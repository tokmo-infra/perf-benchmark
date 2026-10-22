use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::{body::Incoming, service::service_fn, Request, Response};
use hyper_util::rt::{TokioExecutor, TokioIo};
use perf_rusthttp::{http::Http, Command, Content, DecodeError, Decoder};
use rustls::pki_types::{pem::PemObject, CertificateDer, PrivateKeyDer};
use std::{
    convert::Infallible,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

struct Json;
impl Decoder for Json {
    fn feed(&mut self, data: &[u8], _: bool) -> Result<(i64, bool), DecodeError> {
        assert_eq!(data, b"{}");
        Ok((0, true))
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
async fn alpn_http2_reuses_connection_and_releases_uploads() {
    let directory = std::env::temp_dir().join(format!("perf-h2-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let cert = directory.join("cert.pem");
    let key = directory.join("key.pem");
    // Explicit v3 extensions also work with macOS LibreSSL's older defaults.
    let certificate_config = directory.join("openssl.cnf");
    std::fs::write(
        &certificate_config,
        "[req]\ndistinguished_name=dn\nx509_extensions=v3\n[dn]\nCN=localhost\n\
         [v3]\nbasicConstraints=critical,CA:FALSE\nsubjectAltName=DNS:localhost,IP:127.0.0.1\n",
    )
    .unwrap();
    let generated = std::process::Command::new("openssl")
        .args([
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-sha256",
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-keyout",
        ])
        .arg(&key)
        .arg("-config")
        .arg(&certificate_config)
        .arg("-out")
        .arg(&cert)
        .output()
        .unwrap();
    assert!(generated.status.success());
    let mut config = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .unwrap()
    .with_no_client_auth()
    .with_single_cert(
        vec![CertificateDer::from_pem_file(&cert).unwrap()],
        PrivateKeyDer::from_pem_file(&key).unwrap(),
    )
    .unwrap();
    config.alpn_protocols = vec![b"h2".to_vec()];
    let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(config));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let seen = Arc::new(AtomicUsize::new(0));
    let received = seen.clone();
    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let tls = acceptor.accept(socket).await.unwrap();
        let service = service_fn(move |req: Request<Incoming>| {
            let seen = received.clone();
            async move {
                assert_eq!(req.version(), http::Version::HTTP_2);
                assert_eq!(
                    req.into_body().collect().await.unwrap().to_bytes(),
                    b"abcdef"[..]
                );
                seen.fetch_add(1, Ordering::Relaxed);
                Ok::<_, Infallible>(Response::new(Full::new(Bytes::from_static(b"{}"))))
            }
        });
        let _ = hyper::server::conn::http2::Builder::new(TokioExecutor::new())
            .serve_connection(TokioIo::new(tls), service)
            .await;
    });
    let http = Http::default();
    let command = Command {
        url: format!("https://{addr}/"),
        method: "POST".into(),
        session_id: "s".into(),
        connection_scope: "session".into(),
        body_length: 6,
        timeout: 3.,
        ..Default::default()
    };
    for _ in 0..2 {
        let (tx, rx) = tokio::sync::oneshot::channel();
        http.execute(
            &command,
            vec![Bytes::from_static(b"abc"), Bytes::from_static(b"def")],
            Box::new(Json),
            CancellationToken::new(),
            None,
            tx,
        )
        .await;
        let result = rx.await.unwrap();
        assert_eq!(result.status, "succeeded", "{:?}", result.error);
    }
    assert_eq!(seen.load(Ordering::Relaxed), 2);
    http.close_session("s").await;
    http.close().await;
    tokio::time::timeout(Duration::from_secs(2), server)
        .await
        .unwrap()
        .unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}
