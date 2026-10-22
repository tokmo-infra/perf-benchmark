use perf_rusthttp::{http::Http, now, Command, Content, DecodeError, Decoder};
use std::time::Duration;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};
use tokio_util::sync::CancellationToken;
struct Slow;
impl Decoder for Slow {
    fn feed(&mut self, data: &[u8], _: bool) -> Result<(i64, bool), DecodeError> {
        std::thread::sleep(Duration::from_millis(20));
        Ok((i64::from(data != b"[DONE]"), data == b"[DONE]"))
    }
    fn finish(&mut self, _: &str) -> (Content, Option<DecodeError>) {
        (
            Content {
                valid: true,
                ..Default::default()
            },
            None,
        )
    }
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn parsing_and_http_cleanup_do_not_shift_measurements() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        for _ in 0..2 {
            let mut headers = Vec::new();
            while !headers.ends_with(b"\r\n\r\n") {
                headers.push(socket.read_u8().await.unwrap());
            }
            let data = b"data: x\n\ndata: [DONE]\n\n";
            socket
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n",
                        data.len()
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
            socket.write_all(data).await.unwrap();
            socket.write_all(b"\r\n").await.unwrap();
            tokio::time::sleep(Duration::from_millis(60)).await;
            socket.write_all(b"0\r\n\r\n").await.unwrap();
        }
    });
    let http = Http::default();
    for _ in 0..2 {
        let cmd = Command {
            url: format!("http://{address}"),
            method: "POST".into(),
            connection_scope: "trial".into(),
            timeout: 2.,
            stream: true,
            ..Default::default()
        };
        let (tx, rx) = tokio::sync::oneshot::channel();
        http.execute(
            &cmd,
            vec![],
            Box::new(Slow),
            CancellationToken::new(),
            None,
            tx,
        )
        .await;
        let result = rx.await.unwrap();
        assert_eq!(result.status, "succeeded");
        assert_eq!(result.samples.len(), 1);
        assert_eq!(result.completed_ns, Some(result.samples[0][1]));
        assert!(now() - result.completed_ns.unwrap() >= 35_000_000);
    }
    server.await.unwrap();
    http.close().await;
}
#[tokio::test]
async fn expired_request_has_no_start_notification() {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let cmd = Command {
        url: "http://127.0.0.1:1".into(),
        method: "POST".into(),
        connection_scope: "trial".into(),
        timeout: 1e-12,
        ..Default::default()
    };
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    Http::default()
        .execute(
            &cmd,
            vec![],
            Box::new(Slow),
            CancellationToken::new(),
            Some(tx),
            result_tx,
        )
        .await;
    let result = result_rx.await.unwrap();
    assert_eq!(result.error.as_deref(), Some("TimeoutError"));
    assert!(result.started_ns.is_none());
    assert!(rx.await.is_err());
}

async fn execute(http: &Http, url: String, scope: &str) -> perf_rusthttp::types::Outcome {
    let cmd = Command {
        url,
        method: "POST".into(),
        session_id: "idle-session".into(),
        connection_scope: scope.into(),
        timeout: 2.,
        stream: true,
        ..Default::default()
    };
    let (tx, rx) = tokio::sync::oneshot::channel();
    http.execute(
        &cmd,
        vec![],
        Box::new(Slow),
        CancellationToken::new(),
        None,
        tx,
    )
    .await;
    rx.await.unwrap()
}

#[tokio::test]
async fn session_and_trial_reconnect_idle_sockets_without_replaying_requests() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/", listener.local_addr().unwrap());
    let accepts = Arc::new(AtomicUsize::new(0));
    let requests = Arc::new(AtomicUsize::new(0));
    let server_accepts = accepts.clone();
    let server_requests = requests.clone();
    let server = tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            server_accepts.fetch_add(1, Ordering::SeqCst);
            let requests = server_requests.clone();
            tokio::spawn(async move {
                loop {
                    let mut headers = Vec::new();
                    while !headers.ends_with(b"\r\n\r\n") {
                        match socket.read_u8().await {
                            Ok(byte) => headers.push(byte),
                            Err(_) => return,
                        }
                    }
                    requests.fetch_add(1, Ordering::SeqCst);
                    let data = b"data: x\n\ndata: [DONE]\n\n";
                    let response =
                        format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", data.len());
                    socket.write_all(response.as_bytes()).await.unwrap();
                    socket.write_all(data).await.unwrap();
                }
            });
        }
    });
    let session = Http::default();
    let trial = Http::default();
    for http in [&session, &trial] {
        let scope = if std::ptr::eq(http, &session) {
            "session"
        } else {
            "trial"
        };
        for _ in 0..2 {
            assert_eq!(execute(http, url.clone(), scope).await.status, "succeeded");
        }
    }
    assert_eq!(accepts.load(Ordering::SeqCst), 2);
    tokio::time::sleep(Duration::from_millis(4100)).await;
    for (http, scope) in [(&session, "session"), (&trial, "trial")] {
        assert_eq!(execute(http, url.clone(), scope).await.status, "succeeded");
        http.close().await;
    }
    assert_eq!(accepts.load(Ordering::SeqCst), 4);
    assert_eq!(requests.load(Ordering::SeqCst), 6);
    server.abort();
}

#[tokio::test]
async fn peer_disconnect_preserves_transport_stage_and_underlying_error() {
    for body_started in [false, true] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut headers = Vec::new();
            while !headers.ends_with(b"\r\n\r\n") {
                headers.push(socket.read_u8().await.unwrap());
            }
            if body_started {
                socket
                    .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nx")
                    .await
                    .unwrap();
            }
        });
        let http = Http::default();
        let result = execute(&http, url, "session").await;
        server.await.unwrap();
        assert_eq!(result.status, "failed");
        assert_eq!(result.error_origin, Some("transport"));
        assert!(result.samples.is_empty());
        let error = result.error.unwrap();
        let phase = if body_started {
            "response body"
        } else {
            "connect/send"
        };
        assert!(
            error.starts_with(&format!("TransportError: {phase}: ")),
            "{error}"
        );
        assert!(error.contains("connection"), "{error}");
        assert_eq!(result.http_status, body_started.then_some(200));
        http.close().await;
    }
}
