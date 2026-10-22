use crate::transport::Pool;
use crate::{
    now,
    sse::Framer,
    types::{Command, Outcome},
    DecodeError, Decoder,
};
use base64::Engine;
use bytes::Bytes;
use http::{header, HeaderMap, Method};
use std::sync::Arc;
use std::{collections::HashMap, time::Duration};
use tokio::{
    sync::{oneshot, Mutex},
    time::{sleep_until, Instant},
};
use tokio_util::sync::CancellationToken;
use url::Url;

#[derive(Default)]
pub struct Http {
    clients: Mutex<HashMap<(String, bool), Arc<Pool>>>,
}
impl Http {
    pub async fn close_session(&self, id: &str) {
        let closing = {
            let mut clients = self.clients.lock().await;
            [false, true]
                .into_iter()
                .filter_map(|verify| clients.remove(&(id.to_owned(), verify)))
                .collect::<Vec<_>>()
        };
        for client in closing {
            client.close().await;
        }
    }
    pub async fn close(&self) {
        let clients = std::mem::take(&mut *self.clients.lock().await);
        for (_, client) in clients {
            client.close().await;
        }
    }
    async fn client(&self, cmd: &Command) -> std::io::Result<Arc<Pool>> {
        let key = (
            if cmd.connection_scope == "session" {
                cmd.session_id.clone()
            } else {
                String::new()
            },
            cmd.verify_ssl,
        );
        let mut clients = self.clients.lock().await;
        if cmd.connection_scope != "request" {
            if let Some(client) = clients.get(&key) {
                return Ok(client.clone());
            }
        }
        let client = Arc::new(Pool::new(cmd.verify_ssl, &cmd.connection_scope)?);
        if cmd.connection_scope != "request" {
            clients.insert(key, client.clone());
        }
        Ok(client)
    }
    pub async fn execute(
        &self,
        cmd: &Command,
        body: Vec<Bytes>,
        mut decoder: Box<dyn Decoder>,
        cancel: CancellationToken,
        started: Option<oneshot::Sender<i64>>,
        result: oneshot::Sender<Outcome>,
    ) {
        let mut out = Outcome {
            status: "succeeded",
            continuation_valid: true,
            samples: Vec::with_capacity(64),
            ..Default::default()
        };
        if !cmd.timeout.is_finite() || cmd.timeout <= 0.0 {
            fail(&mut out, "transport", "TimeoutError", true);
            out.completed_ns = Some(now());
            let _ = result.send(out);
            return;
        }
        let deadline = Instant::now() + Duration::from_secs_f64(cmd.timeout);
        let mut response = None;
        let mut connection = None;
        let mut pool = None;
        let mut decode_error = None;
        let mut scenario_at = None;
        let mut scenario_deadline = None;
        let work = async {
            if !matches!(
                cmd.connection_scope.as_str(),
                "request" | "session" | "trial"
            ) || (cmd.connection_scope == "session" && cmd.session_id.is_empty())
            {
                fail(&mut out, "tool", "InvalidConnectionScope", false);
                return;
            }
            let client = match self.client(cmd).await {
                Ok(v) => v,
                Err(_) => {
                    fail(&mut out, "transport", "InvalidRequest", false);
                    return;
                }
            };
            let parsed = (|| {
                let mut url = Url::parse(&cmd.url).map_err(std::io::Error::other)?;
                if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
                    return Err(std::io::Error::other("invalid URL"));
                }
                url.set_fragment(None);
                let method =
                    Method::from_bytes(cmd.method.as_bytes()).map_err(std::io::Error::other)?;
                let mut headers = HeaderMap::new();
                for (key, value) in &cmd.headers {
                    headers.insert(
                        header::HeaderName::from_bytes(key.as_bytes())
                            .map_err(std::io::Error::other)?,
                        value.parse().map_err(std::io::Error::other)?,
                    );
                }
                if !url.username().is_empty() || url.password().is_some() {
                    let user =
                        percent_encoding::percent_decode_str(url.username()).decode_utf8_lossy();
                    let password =
                        percent_encoding::percent_decode_str(url.password().unwrap_or(""))
                            .decode_utf8_lossy();
                    let credentials = base64::engine::general_purpose::STANDARD
                        .encode(format!("{user}:{password}"));
                    headers.entry(header::AUTHORIZATION).or_insert(
                        format!("Basic {credentials}")
                            .parse()
                            .map_err(std::io::Error::other)?,
                    );
                    let _ = url.set_username("");
                    let _ = url.set_password(None);
                }
                if !headers.contains_key(header::RANGE) {
                    headers
                        .entry(header::ACCEPT_ENCODING)
                        .or_insert(http::HeaderValue::from_static("gzip"));
                }
                headers
                    .entry(header::ACCEPT)
                    .or_insert(http::HeaderValue::from_static("*/*"));
                headers.entry(header::CONTENT_LENGTH).or_insert(
                    cmd.body_length
                        .to_string()
                        .parse()
                        .map_err(std::io::Error::other)?,
                );
                Ok::<_, std::io::Error>((url, method, headers))
            })();
            let (mut url, mut method, mut headers) = match parsed {
                Ok(v) => v,
                Err(_) => {
                    fail(&mut out, "transport", "InvalidRequest", false);
                    return;
                }
            };
            pool = Some(client.clone());
            if cancel.is_cancelled() || Instant::now() >= deadline {
                return;
            }
            let at = now();
            out.started_ns = Some(at);
            if let Some(sender) = started {
                let _ = sender.send(at);
            }
            let mut request_body = body;
            let mut redirects = 0;
            let received = loop {
                let sent = async {
                    connection = Some(client.acquire(&url).await?);
                    connection
                        .as_mut()
                        .unwrap()
                        .send(&url, method.clone(), headers.clone(), &request_body)
                        .await
                }
                .await;
                let mut received = match sent {
                    Ok(v) => v,
                    Err(error) => {
                        transport_failure(&mut out, "connect/send", &error);
                        out.completed_ns = Some(now());
                        return;
                    }
                };
                let status = received.status().as_u16();
                if !matches!(status, 301 | 302 | 303 | 307 | 308) {
                    break received;
                }
                let next = received
                    .headers
                    .get(header::LOCATION)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| url.join(v).ok());
                let Some(mut next) = next else {
                    break received;
                };
                redirects += 1;
                if redirects >= 10 || !matches!(next.scheme(), "http" | "https") {
                    fail(&mut out, "transport", "TransportError", true);
                    out.completed_ns = Some(now());
                    return;
                }
                next.set_fragment(None);
                if next.host_str() != url.host_str()
                    || next.port_or_known_default() != url.port_or_known_default()
                {
                    for name in [
                        header::AUTHORIZATION,
                        header::COOKIE,
                        header::PROXY_AUTHORIZATION,
                        header::WWW_AUTHENTICATE,
                    ] {
                        headers.remove(name);
                    }
                }
                headers.remove(header::HOST);
                if !(url.scheme() == "https" && next.scheme() == "http") {
                    let mut referer = url.clone();
                    let _ = referer.set_username("");
                    let _ = referer.set_password(None);
                    if let Ok(value) = referer.as_str().parse() {
                        headers.insert(header::REFERER, value);
                    }
                } else {
                    headers.remove(header::REFERER);
                }
                if ((status == 301 || status == 302) && method == Method::POST)
                    || (status == 303 && method != Method::GET && method != Method::HEAD)
                {
                    method = Method::GET;
                    request_body.clear();
                    for name in [
                        header::CONTENT_TYPE,
                        header::CONTENT_LENGTH,
                        header::TRANSFER_ENCODING,
                        header::CONTENT_ENCODING,
                    ] {
                        headers.remove(name);
                    }
                }
                let reusable = tokio::time::timeout(Duration::from_millis(100), async {
                    if !received.drain().await {
                        return false;
                    }
                    connection.as_ref().unwrap().upload_finished().await;
                    true
                })
                .await
                .unwrap_or(false);
                let previous = connection.take().unwrap();
                if reusable {
                    client.put(previous).await;
                } else {
                    previous.close().await;
                }
                url = next;
            };
            let status = received.status().as_u16();
            out.http_status = Some(status);
            response = Some(received);
            if status >= 400 {
                fail(
                    &mut out,
                    "service",
                    &format!("HTTP {status}: upstream request rejected"),
                    matches!(status, 429 | 502 | 503 | 504),
                );
                out.completed_ns = Some(now());
                return;
            }
            let mut framer = Framer::default();
            let mut json = Vec::new();
            let mut last = None;
            'reading: loop {
                let received = tokio::select! {
                    biased;
                    _ = async {
                        if let Some((deadline, _)) = scenario_deadline {
                            sleep_until(deadline).await
                        } else {
                            std::future::pending::<()>().await
                        }
                    } => {
                        scenario_at = scenario_deadline.map(|(_, at)| at);
                        break;
                    },
                    value = response.as_mut().unwrap().chunk() => value,
                };
                // Freeze application-read time before framing, decoding, or dispatch.
                let at = now();
                let data = match received {
                    Ok(Some(v)) if !v.is_empty() => v,
                    Ok(Some(_)) => continue,
                    Ok(None) => break,
                    Err(error) => {
                        transport_failure(&mut out, "response body", &error);
                        last = Some(at);
                        break;
                    }
                };
                last = Some(at);
                if !cmd.stream {
                    json.extend_from_slice(&data);
                    continue;
                }
                for &byte in data.as_ref() {
                    let event = match framer.push(byte) {
                        Ok(Some(v)) if !v.is_empty() => v,
                        Ok(_) => continue,
                        Err(e) => {
                            decode_error = Some(e);
                            break 'reading;
                        }
                    };
                    match decoder.feed(event.as_bytes(), true) {
                        Ok((chars, done)) => {
                            if chars > 0 {
                                let count = out.samples.len() as i64 + 1;
                                out.samples.push([count, at, chars]);
                                if cmd.plan.after_outputs.is_some_and(|n| count >= n) {
                                    scenario_at = Some(at);
                                    break 'reading;
                                }
                                if count == 1 {
                                    if let Some(ms) = cmd.plan.after_first_output_ms {
                                        let trigger = at + (ms * 1e6) as i64;
                                        scenario_deadline = Some((
                                            Instant::now()
                                                + Duration::from_nanos(
                                                    (trigger - now()).max(0) as u64
                                                ),
                                            trigger,
                                        ));
                                    }
                                }
                            }
                            if done {
                                break 'reading;
                            }
                        }
                        Err(e) => {
                            decode_error = Some(e);
                            break 'reading;
                        }
                    }
                }
            }
            out.completed_ns = Some(last.unwrap_or_else(now));
            if !cmd.stream && out.error_origin != Some("transport") {
                decode_error = decoder.feed(&json, false).err();
            }
        };
        enum End {
            Done,
            Cancel,
            Timeout,
        }
        let end = tokio::select! {
            biased;
            _ = cancel.cancelled() => End::Cancel,
            _ = sleep_until(deadline) => End::Timeout,
            _ = work => End::Done,
        };
        if matches!(end, End::Cancel) || cancel.is_cancelled() {
            cancelled(&mut out, "client", "cancelled", now(), false);
        } else if let Some(at) = scenario_at {
            let count_reached = cmd
                .plan
                .after_outputs
                .is_some_and(|n| out.samples.len() as i64 >= n);
            cancelled(
                &mut out,
                "scenario",
                if count_reached {
                    "output count reached"
                } else {
                    "output delay reached"
                },
                at,
                cmd.plan.action == "fail",
            );
        } else if matches!(end, End::Timeout)
            || Instant::now() >= deadline
            || (out.started_ns.is_none() && out.error.is_none())
        {
            fail(&mut out, "transport", "TimeoutError", true);
            out.completed_ns = Some(now());
        }
        let (content, finish_error) = decoder.finish(out.status);
        out.output_chars = content.chars;
        out.usage = content.usage;
        out.output_kind = if content.kind.is_empty() {
            "text"
        } else {
            content.kind
        };
        out.continuation = content.continuation;
        out.continuation_valid = content.valid;
        out.finish_reason = if content.finish.is_empty() {
            None
        } else {
            Some(content.finish)
        };
        if out.error_origin != Some("transport") && out.status != "cancelled" {
            if let Some(error) = decode_error.or(finish_error) {
                let incomplete = matches!(error, DecodeError::Incomplete);
                fail(
                    &mut out,
                    "protocol",
                    if incomplete {
                        "IncompleteResponse"
                    } else {
                        "ProtocolError"
                    },
                    incomplete,
                );
            }
        }
        // Facts are frozen before bounded reuse cleanup or connection task shutdown.
        let succeeded = out.status == "succeeded";
        let _ = result.send(out);
        let mut reusable = false;
        if succeeded {
            if let Some(response) = response.as_mut() {
                let cleanup_deadline = deadline.min(Instant::now() + Duration::from_millis(100));
                reusable = tokio::select! {
                    _ = cancel.cancelled() => false,
                    _ = sleep_until(cleanup_deadline) => false,
                    value = async {
                        if !response.drain().await { return false; }
                        if let Some(connection) = connection.as_ref() { connection.upload_finished().await; }
                        true
                    } => value,
                };
            }
        }
        drop(response);
        if let Some(connection) = connection {
            if reusable {
                pool.unwrap().put(connection).await;
            } else {
                connection.close().await;
            }
        }
    }
}

fn transport_failure(out: &mut Outcome, phase: &str, error: &std::io::Error) {
    let mut message = format!("TransportError: {phase}: {error}");
    let mut previous = error.to_string();
    let mut source = std::error::Error::source(error);
    while let Some(cause) = source {
        let detail = cause.to_string();
        if detail != previous {
            message.push_str(": ");
            message.push_str(&detail);
        }
        previous = detail;
        source = cause.source();
    }
    fail(out, "transport", &message, true);
}

fn fail(out: &mut Outcome, origin: &'static str, error: &str, retryable: bool) {
    out.status = "failed";
    out.error_origin = Some(origin);
    out.error = Some(error.to_owned());
    out.retryable = retryable;
}
fn cancelled(
    out: &mut Outcome,
    origin: &'static str,
    reason: &str,
    trigger: i64,
    fail_action: bool,
) {
    out.status = if fail_action { "failed" } else { "cancelled" };
    out.error_origin = Some(origin);
    out.error = Some(reason.into());
    out.retryable = fail_action;
    out.trigger_ns = Some(trigger);
    out.decision_ns = Some(now());
    out.applied_ns = Some(now());
    out.completed_ns = out.applied_ns;
}
