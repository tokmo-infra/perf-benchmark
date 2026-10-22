//! Hyper connection leases: each Attempt can abort and join its own connection.
use bytes::Bytes;
use futures_util::{Stream, StreamExt, TryStreamExt};
use http::{header, HeaderMap, Method, Request, Uri};
use http_body_util::{combinators::UnsyncBoxBody, BodyExt, StreamBody};
use hyper::{body::Frame, client::conn};
use hyper_util::{
    client::proxy::matcher::Matcher,
    rt::{TokioExecutor, TokioIo},
};
use rustls::{
    client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier},
    pki_types::{CertificateDer, ServerName, UnixTime},
    DigitallySignedStruct, SignatureScheme,
};
use std::{
    collections::HashMap,
    io,
    pin::Pin,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::{
    io::{AsyncRead, AsyncWrite},
    net::TcpStream,
    sync::{Mutex, Notify},
    task::JoinHandle,
};
use tokio_rustls::TlsConnector;
use tokio_util::io::{ReaderStream, StreamReader};
use url::Url;

type HttpBody = UnsyncBoxBody<Bytes, io::Error>;
type BodyStream = Pin<Box<dyn Stream<Item = io::Result<Bytes>> + Send>>;
trait Socket: AsyncRead + AsyncWrite + Unpin + Send {}
impl<T: AsyncRead + AsyncWrite + Unpin + Send> Socket for T {}
type Io = Box<dyn Socket>;
fn error(value: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> io::Error {
    io::Error::other(value)
}

#[derive(Debug)]
struct Unverified;
impl ServerCertVerifier for Unverified {
    fn verify_server_cert(
        &self,
        _: &CertificateDer<'_>,
        _: &[CertificateDer<'_>],
        _: &ServerName<'_>,
        _: &[u8],
        _: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }
    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        signature: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            signature,
            &rustls::crypto::ring::default_provider().signature_verification_algorithms,
        )
    }
    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        signature: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            signature,
            &rustls::crypto::ring::default_provider().signature_verification_algorithms,
        )
    }
    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}
pub struct Pool {
    tls: Arc<rustls::ClientConfig>,
    proxy: Matcher,
    idle: Mutex<HashMap<String, Vec<Connection>>>,
    request_scope: bool,
    session_scope: bool,
}
impl Pool {
    pub fn new(verify: bool, scope: &str) -> io::Result<Self> {
        let builder = rustls::ClientConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .map_err(error)?;
        let mut tls = if verify {
            let mut roots = rustls::RootCertStore::empty();
            for cert in rustls_native_certs::load_native_certs().certs {
                roots.add(cert).map_err(error)?;
            }
            builder.with_root_certificates(roots).with_no_client_auth()
        } else {
            builder
                .dangerous()
                .with_custom_certificate_verifier(Arc::new(Unverified))
                .with_no_client_auth()
        };
        tls.alpn_protocols = if scope == "request" {
            vec![b"http/1.1".to_vec()]
        } else {
            vec![b"h2".to_vec(), b"http/1.1".to_vec()]
        };
        Ok(Self {
            tls: Arc::new(tls),
            proxy: Matcher::from_env(),
            idle: Mutex::new(HashMap::new()),
            request_scope: scope == "request",
            session_scope: scope == "session",
        })
    }
    pub async fn acquire(&self, url: &Url) -> io::Result<Connection> {
        let key = url.origin().ascii_serialization();
        loop {
            let cached = self.idle.lock().await.get_mut(&key).and_then(Vec::pop);
            match cached {
                Some(mut connection) => {
                    // Reconnect before the common five-second server idle timeout.
                    // Session ownership does not make an idle socket stay alive.
                    if connection.idle_at.elapsed() < Duration::from_secs(4)
                        && connection.ready().await.is_ok()
                    {
                        return Ok(connection);
                    }
                    connection.close().await;
                }
                None => break,
            }
        }
        tokio::time::timeout(Duration::from_secs(30), self.connect(url))
            .await
            .map_err(error)?
    }
    async fn tls(&self, io: Io, url: &Url, http2: bool) -> io::Result<(Io, bool)> {
        let name = ServerName::try_from(
            url.host_str()
                .ok_or_else(|| error("missing host"))?
                .to_owned(),
        )
        .map_err(error)?;
        let config = if http2 {
            self.tls.clone()
        } else {
            let mut config = (*self.tls).clone();
            config.alpn_protocols = vec![b"http/1.1".to_vec()];
            Arc::new(config)
        };
        let stream = TlsConnector::from(config).connect(name, io).await?;
        let h2 = stream.get_ref().1.alpn_protocol() == Some(b"h2");
        Ok((Box::new(stream), h2))
    }
    async fn connect(&self, url: &Url) -> io::Result<Connection> {
        let uri: Uri = url.as_str().parse().map_err(error)?;
        let proxy = self.proxy.intercept(&uri);
        let target = proxy
            .as_ref()
            .map(|p| Url::parse(&p.uri().to_string()).map_err(error))
            .transpose()?
            .unwrap_or_else(|| url.clone());
        if !matches!(target.scheme(), "http" | "https") {
            return Err(error("unsupported proxy scheme"));
        }
        let socket = TcpStream::connect((
            target.host_str().ok_or_else(|| error("missing host"))?,
            target
                .port_or_known_default()
                .ok_or_else(|| error("missing port"))?,
        ))
        .await?;
        socket.set_nodelay(true)?;
        let mut io: Io = Box::new(socket);
        let mut h2 = false;
        if target.scheme() == "https" {
            (io, h2) = self.tls(io, &target, proxy.is_none()).await?;
        }
        let mut proxy_auth = None;
        let absolute = proxy.is_some() && url.scheme() == "http";
        if let Some(proxy) = proxy {
            if url.scheme() == "https" {
                let authority = format!(
                    "{}:{}",
                    url.host().ok_or_else(|| error("missing host"))?,
                    url.port_or_known_default().unwrap()
                );
                let (mut sender, driver) = conn::http1::handshake::<_, HttpBody>(TokioIo::new(io))
                    .await
                    .map_err(error)?;
                let driver = tokio_util::task::AbortOnDropHandle::new(tokio::spawn(async move {
                    driver.with_upgrades().await
                }));
                let mut request = Request::builder()
                    .method(Method::CONNECT)
                    .uri(&authority)
                    .header(header::HOST, &authority);
                if let Some(auth) = proxy.basic_auth() {
                    request = request.header(header::PROXY_AUTHORIZATION, auth);
                }
                let mut response = sender
                    .send_request(request.body(body(&[]).0).map_err(error)?)
                    .await
                    .map_err(error)?;
                if !response.status().is_success() {
                    return Err(error("proxy CONNECT rejected"));
                }
                let upgraded = hyper::upgrade::on(&mut response).await.map_err(error)?;
                driver.await.map_err(error)?.map_err(error)?;
                (io, h2) = self
                    .tls(Box::new(TokioIo::new(upgraded)), url, true)
                    .await?;
            } else {
                proxy_auth = proxy.basic_auth().cloned();
            }
        }
        let (sender, driver) = if h2 {
            let (sender, driver) = conn::http2::handshake(TokioExecutor::new(), TokioIo::new(io))
                .await
                .map_err(error)?;
            (
                Sender::H2(sender),
                tokio::spawn(async move {
                    let _ = driver.await;
                }),
            )
        } else {
            let (sender, driver) = conn::http1::handshake(TokioIo::new(io))
                .await
                .map_err(error)?;
            (
                Sender::H1(sender),
                tokio::spawn(async move {
                    let _ = driver.await;
                }),
            )
        };
        Ok(Connection {
            sender,
            driver,
            key: url.origin().ascii_serialization(),
            absolute,
            proxy_auth,
            upload: Arc::new(Upload::default()),
            idle_at: Instant::now(),
        })
    }
    pub async fn put(&self, mut connection: Connection) {
        if self.request_scope || connection.driver.is_finished() {
            connection.close().await;
            return;
        }
        connection.idle_at = Instant::now();
        let mut idle = self.idle.lock().await;
        let entries = idle.entry(connection.key.clone()).or_default();
        if entries.len() < if self.session_scope { 1 } else { 10000 } {
            entries.push(connection);
        } else {
            drop(idle);
            connection.close().await;
        }
    }
    pub async fn close(&self) {
        let connections = std::mem::take(&mut *self.idle.lock().await);
        for (_, entries) in connections {
            for connection in entries {
                connection.close().await;
            }
        }
    }
}
enum Sender {
    H1(conn::http1::SendRequest<HttpBody>),
    H2(conn::http2::SendRequest<HttpBody>),
}
pub struct Connection {
    sender: Sender,
    driver: JoinHandle<()>,
    key: String,
    absolute: bool,
    proxy_auth: Option<http::HeaderValue>,
    idle_at: Instant,
    upload: Arc<Upload>,
}
impl Drop for Connection {
    fn drop(&mut self) {
        self.driver.abort();
    }
}
impl Connection {
    async fn ready(&mut self) -> io::Result<()> {
        match &mut self.sender {
            Sender::H1(s) => s.ready().await,
            Sender::H2(s) => s.ready().await,
        }
        .map_err(error)
    }
    pub async fn send(
        &mut self,
        url: &Url,
        method: Method,
        mut headers: HeaderMap,
        segments: &[Bytes],
    ) -> io::Result<Response> {
        let authority = url[url::Position::BeforeHost..url::Position::AfterPort].to_owned();
        let target = if self.absolute || matches!(self.sender, Sender::H2(_)) {
            url.as_str().to_owned()
        } else {
            url[url::Position::BeforePath..url::Position::AfterQuery].to_owned()
        };
        headers
            .entry(header::HOST)
            .or_insert(authority.parse().map_err(error)?);
        if let Some(auth) = &self.proxy_auth {
            headers
                .entry(header::PROXY_AUTHORIZATION)
                .or_insert(auth.clone());
        }
        let (body, upload) = body(segments);
        self.upload = upload;
        let mut request = Request::builder()
            .method(method)
            .uri(target)
            .body(body)
            .map_err(error)?;
        *request.headers_mut() = headers;
        let response = match &mut self.sender {
            Sender::H1(s) => s.send_request(request).await,
            Sender::H2(s) => s.send_request(request).await,
        }
        .map_err(error)?;
        let (parts, incoming) = response.into_parts();
        let stream = incoming.into_data_stream().map_err(error);
        let data: BodyStream = if parts
            .headers
            .get(header::CONTENT_ENCODING)
            .is_some_and(|v| v == "gzip")
        {
            Box::pin(ReaderStream::new(
                async_compression::tokio::bufread::GzipDecoder::new(StreamReader::new(stream)),
            ))
        } else {
            Box::pin(stream)
        };
        Ok(Response {
            status: parts.status,
            headers: parts.headers,
            data,
            eof: false,
        })
    }
    pub async fn upload_finished(&self) {
        loop {
            let changed = self.upload.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if self.upload.count.load(Ordering::Acquire) == 0 {
                return;
            }
            changed.await;
        }
    }
    pub async fn close(mut self) {
        self.driver.abort();
        let _ = (&mut self.driver).await;
    }
}
#[derive(Default)]
struct Upload {
    count: AtomicUsize,
    changed: Notify,
}
struct Ticket(Arc<Upload>);
impl Ticket {
    fn new(upload: &Arc<Upload>) -> Self {
        upload.count.fetch_add(1, Ordering::Relaxed);
        Self(upload.clone())
    }
}
impl Drop for Ticket {
    fn drop(&mut self) {
        self.0.count.fetch_sub(1, Ordering::Release);
        self.0.changed.notify_one();
    }
}
struct Sending {
    data: Bytes,
    _ticket: Ticket,
}
impl AsRef<[u8]> for Sending {
    fn as_ref(&self) -> &[u8] {
        &self.data
    }
}
fn body(segments: &[Bytes]) -> (HttpBody, Arc<Upload>) {
    let upload = Arc::new(Upload::default());
    let mut source = std::collections::VecDeque::from(segments.to_vec());
    let tracker = upload.clone();
    let mut stream_ticket = Some(Ticket::new(&upload));
    let frames = std::iter::from_fn(move || {
        while source.front().is_some_and(Bytes::is_empty) {
            source.pop_front();
        }
        let Some(front) = source.front_mut() else {
            stream_ticket.take();
            return None;
        };
        let data = front.split_to(front.len().min(65536));
        if front.is_empty() {
            source.pop_front();
        }
        Some(Ok(Frame::data(Bytes::from_owner(Sending {
            data,
            _ticket: Ticket::new(&tracker),
        }))))
    });
    (
        StreamBody::new(futures_util::stream::iter(frames)).boxed_unsync(),
        upload,
    )
}
pub struct Response {
    pub status: http::StatusCode,
    pub headers: HeaderMap,
    data: BodyStream,
    pub eof: bool,
}
impl Response {
    pub fn status(&self) -> http::StatusCode {
        self.status
    }
    pub async fn chunk(&mut self) -> io::Result<Option<Bytes>> {
        match self.data.next().await {
            Some(value) => value.map(Some),
            None => {
                self.eof = true;
                Ok(None)
            }
        }
    }
    pub async fn drain(&mut self) -> bool {
        let mut total = 0;
        while total < 65536 {
            match self.chunk().await {
                Ok(Some(bytes)) => total += bytes.len(),
                Ok(None) => return true,
                Err(_) => return false,
            }
        }
        false
    }
}
