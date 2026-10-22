"""Transport policies exercised through the production native binding."""
import asyncio
import gzip

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from perf_benchmark import _native
from tests.runtime.native_support import request
from perf_benchmark.runtime.http.request import PreparedBody


@pytest.mark.parametrize('status', [301, 302, 303, 307, 308])
async def test_redirect_preserves_replayable_body(status):
    received = []
    async def redirect(req):
        await req.read()
        return web.Response(status=status, headers={'Location': '/final'})
    async def final(req):
        received.append((req.method, await req.read(), req.headers.get('Referer')))
        return web.json_response({'choices': [{'message': {'content': 'ok'}}]})
    app = web.Application()
    app.router.add_post('/', redirect)
    app.router.add_route('*', '/final', final)
    native = _native.create_trial('openai-chat')
    async with TestServer(app) as server:
        try:
            value = await request(native, dict(url=str(server.make_url('/')), body_length=6), PreparedBody((b'abc', memoryview(b'def')), 6))
            assert value.status == 'succeeded'
            assert received == [('POST' if status in (307, 308) else 'GET', b'abcdef' if status in (307, 308) else b'', str(server.make_url('/')))]
        finally:
            await native.close()


async def test_gzip_response_is_decoded_before_protocol():
    async def complete(req):
        assert req.headers['Accept-Encoding'] == 'gzip'
        data = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n\n'
        return web.Response(body=gzip.compress(data), headers={'Content-Encoding': 'gzip'})
    app = web.Application()
    app.router.add_post('/', complete)
    async with TestServer(app) as server:
        native = _native.create_trial('openai-chat')
        try:
            result = await request(native, dict(url=str(server.make_url('/')), stream=True))
            assert result.status == 'succeeded' and result.output_chars == 5
        finally:
            await native.close()


async def test_http_proxy_and_no_proxy(monkeypatch):
    seen = []
    async def proxy(reader, writer):
        data = await reader.readuntil(b'\r\n\r\n')
        seen.append(data)
        body = b'{"choices":[{"message":{"content":"ok"}}]}'
        writer.write(f'HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n'.encode() + body)
        await writer.drain()
        writer.close()
        await writer.wait_closed()
    server = await asyncio.start_server(proxy, '127.0.0.1', 0)
    port = server.sockets[0].getsockname()[1]
    for key in ('http_proxy', 'HTTP_PROXY', 'all_proxy', 'ALL_PROXY', 'no_proxy', 'NO_PROXY'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('http_proxy', f'http://user:secret@127.0.0.1:{port}')
    native = _native.create_trial('openai-chat')
    try:
        value = await request(native, dict(url='http://does-not-exist.invalid/v1/test'))
        assert value.status == 'succeeded'
        assert seen[0].startswith(b'POST http://does-not-exist.invalid/v1/test HTTP/1.1')
        assert b'proxy-authorization: Basic dXNlcjpzZWNyZXQ=' in seen[0]
    finally:
        await native.close()
    monkeypatch.setenv('no_proxy', '127.0.0.1')
    native = _native.create_trial('openai-chat')
    try:
        value = await request(native, dict(url='http://127.0.0.1:1'))
        assert value.status == 'failed'
        assert len(seen) == 1
    finally:
        await native.close()
        server.close()
        await server.wait_closed()


async def test_redirect_strips_cross_origin_credentials():
    seen = []
    async def final(req):
        seen.append(req.headers.get('Authorization'))
        return web.json_response({'choices': [{'message': {'content': 'ok'}}]})
    target = web.Application()
    target.router.add_post('/', final)
    async with TestServer(target) as end:
        async def redirect(req):
            assert req.headers['Authorization'] == 'Basic dXNlcjpzZWNyZXQ='
            return web.Response(status=307, headers={'Location': str(end.make_url('/'))})
        app = web.Application()
        app.router.add_post('/', redirect)
        async with TestServer(app) as start:
            native = _native.create_trial('openai-chat')
            try:
                url = str(start.make_url('/')).replace('http://', 'http://user:secret@')
                result = await request(native, dict(url=url))
                assert result.status == 'succeeded' and seen == [None]
            finally:
                await native.close()


async def test_trusted_tls_and_https_proxy_connect(tmp_path, monkeypatch):
    import ssl
    import subprocess
    cert, key = tmp_path / 'cert.pem', tmp_path / 'key.pem'
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1',
                    '-subj', '/CN=localhost', '-addext', 'basicConstraints=critical,CA:FALSE',
                    '-addext', 'subjectAltName=DNS:localhost,IP:127.0.0.1',
                    '-keyout', str(key), '-out', str(cert)], check=True, capture_output=True)
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(cert, key)
    monkeypatch.setenv('SSL_CERT_FILE', str(cert))
    for name in ('https_proxy', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY', 'no_proxy', 'NO_PROXY'):
        monkeypatch.delenv(name, raising=False)
    async def final(req):
        await req.read()
        return web.json_response({'choices': [{'message': {'content': 'ok'}}]})
    app = web.Application()
    app.router.add_post('/', final)
    destination = TestServer(app, scheme='https')
    await destination.start_server(ssl=tls)
    connections = []
    async def tunnel(reader, writer):
        first = await reader.readuntil(b'\r\n\r\n')
        connections.append(first)
        remote, remote_writer = await asyncio.open_connection('127.0.0.1', destination.port)
        writer.write(b'HTTP/1.1 200 Connection established\r\n\r\n')
        await writer.drain()
        async def copy(source, sink):
            try:
                while data := await source.read(65536):
                    sink.write(data)
                    await sink.drain()
            finally:
                sink.close()
        await asyncio.gather(copy(reader, remote_writer), copy(remote, writer), return_exceptions=True)
    proxy = await asyncio.start_server(tunnel, '127.0.0.1', 0)
    try:
        for proxied in (False, True):
            if proxied:
                monkeypatch.setenv('https_proxy', f'http://127.0.0.1:{proxy.sockets[0].getsockname()[1]}')
            native = _native.create_trial('openai-chat')
            try:
                result = await request(native, dict(url=str(destination.make_url('/')), verify_ssl=True))
                assert result.status == 'succeeded', (proxied, result.error)
            finally:
                await native.close()
        assert len(connections) == 1
        assert connections[0].startswith(b'CONNECT 127.0.0.1:')
    finally:
        await destination.close()
        proxy.close()
        await proxy.wait_closed()
