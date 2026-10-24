"""Resolve default reasoning once, before measured runs and frozen evidence."""
import asyncio
from copy import deepcopy
import json
import logging

import aiohttp

from perf_benchmark.runtime.cancellation import CancellationToken

CANDIDATES = ('max', 'xhigh', 'high')
LIMIT = 64 * 1024
log = logging.getLogger(__name__)


def _selection(extra, protocol):
    if not isinstance(extra, dict):
        raise ValueError('extra_body must be a JSON object')
    key = 'reasoning_effort' if protocol == 'openai-chat' else 'reasoning'
    if key in extra:
        value = extra[key]
        if protocol == 'openai-responses' and value is not None and not isinstance(value, dict):
            raise ValueError('reasoning must be an object or null')
        if protocol == 'openai-responses' and isinstance(value, dict):
            value = value.get('effort', 'auto')
        if value != 'auto':
            if value is None:
                if isinstance(extra[key], dict):
                    extra[key].pop('effort', None)
                else:
                    extra.pop(key)
            elif not isinstance(value, str) or not value.strip():
                raise ValueError('reasoning effort must be a nonempty string, auto or null')
            return 'explicit', value
    template = extra.get('chat_template_kwargs', {})
    switches = [extra[k] for k in ('thinking', 'enable_thinking') if k in extra]
    if isinstance(template, dict):
        switches += [template[k] for k in ('thinking', 'enable_thinking') if k in template]
    enabled = all(value is True or isinstance(value, dict) and (
        value.get('enabled') is True or value.get('type') == 'enabled'
    ) for value in switches)
    # Explicit auto effort composes with enabled switches. A switch alone,
    # or an explicit disabled switch, keeps the caller's existing policy.
    if switches and (key not in extra or not enabled):
        extra.pop(key, None)
        return 'explicit-thinking', None
    if protocol == 'openai-responses':
        extra.setdefault(key, {}).pop('effort', None)
    else:
        extra.pop(key, None)
    return 'auto', None


def _set_effort(extra, protocol, value):
    if protocol == 'openai-chat':
        extra['reasoning_effort'] = value
    else:
        extra.setdefault('reasoning', {})['effort'] = value


def _decode_error(data):
    try:
        payloads = [json.loads(data)]
    except (ValueError, UnicodeError):
        payloads = []
        for line in data.splitlines():
            if line.startswith(b'data:') and line[5:].strip() != b'[DONE]':
                try:
                    payloads.append(json.loads(line[5:]))
                except (ValueError, UnicodeError):
                    continue
    return next((p['error'] for p in payloads if isinstance(p, dict) and p.get('error')), None), payloads


def _unsupported(status, data, effort):
    if status not in (200, 400, 422, 500):
        return False
    error, payloads = _decode_error(data)
    if status == 200 and error is None:
        return False
    if isinstance(error, dict) and error.get('param') not in (
        None, 'reasoning', 'reasoning_effort', 'reasoning.effort', 'effort',
    ):
        return False
    text = json.dumps(error if error else payloads).lower()
    field = (isinstance(error, dict) and error.get('param') == 'reasoning') or any(
        key in text for key in ('reasoning_effort', 'reasoning.effort', 'effort'))
    rejection = any(word in text for word in (
        'unsupported', 'not supported', 'invalid', 'must be', 'should be',
        'allowed', 'permitted', 'literal_error', 'extra_forbidden', 'unrecognized',
        'unknown parameter', 'not one of',
    ))
    return field and rejection


async def _negotiate(config):
    binding = config.get('pipeline', {}).get('executor', {})
    protocol = binding.get('plugin')
    if protocol not in ('openai-chat', 'openai-responses'):
        return config, None
    config = deepcopy(config)
    settings = config['pipeline']['executor'].setdefault('settings', {})
    extra = settings.setdefault('extra_body', {})
    mode, selected = _selection(extra, protocol)
    if mode != 'auto':
        return config, dict(mode=mode, selected=selected, probes=[])
    if protocol == 'openai-chat':
        from perf_benchmark.plugins.executor.openai_chat import OpenAIChatExecutor, validate, reasoning_probe
        plugin = OpenAIChatExecutor(validate(settings))
    else:
        from perf_benchmark.plugins.executor.openai_responses import ResponsesExecutor, validate, reasoning_probe
        plugin = ResponsesExecutor(validate(settings))
    parsed = plugin.settings
    probes = []
    timeout = aiohttp.ClientTimeout(total=min(parsed['timeout_seconds'], 30))
    connector = aiohttp.TCPConnector(ssl=parsed.get('verify_ssl', False))
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as client:
        for effort in CANDIDATES:
            accepted = True
            for endpoint, (url, body) in enumerate(reasoning_probe(parsed, effort)):
                try:
                    async with client.post(url, json=body, headers=plugin.request_headers(parsed),
                                           allow_redirects=False) as response:
                        chunks, size = [], 0
                        while chunk := await response.content.read(8192):
                            size += len(chunk)
                            if size > LIMIT:
                                raise ValueError('reasoning capability response exceeds 64 KiB')
                            chunks.append(chunk)
                        data = b''.join(chunks)
                        status = response.status
                except (aiohttp.ClientError, TimeoutError) as exc:
                    raise ValueError('reasoning capability probe failed: ' + type(exc).__name__) from None
                unsupported = _unsupported(status, data, effort)
                error, payloads = _decode_error(data)
                probes.append(dict(effort=effort, endpoint=endpoint, http_status=status,
                                   accepted=200 <= status < 300 and error is None and not unsupported))
                if unsupported:
                    accepted = False
                    break
                if not 200 <= status < 300 or error is not None or not payloads:
                    raise ValueError(f'reasoning capability probe failed: HTTP {status}; workload unchanged')
            if accepted:
                _set_effort(extra, protocol, effort)
                log.warning('Reasoning effort selected before measurement: %s', effort)
                return config, dict(mode='auto', selected=effort, candidates=list(CANDIDATES),
                                    probes=probes, basis='probe_accepted; model_effect_unverified')
    raise ValueError('service rejected reasoning effort max, xhigh and high; configure an explicit effort or null')


async def resolve(config, cancellation=None):
    """Keep capability probes out of request counts, timings and measurement facts."""
    token = cancellation or CancellationToken()
    if token.is_cancelled():
        raise asyncio.CancelledError(token.reason)
    work = asyncio.create_task(_negotiate(config))
    stopped = asyncio.create_task(token.wait())
    try:
        await asyncio.wait((work, stopped), return_when=asyncio.FIRST_COMPLETED)
        if token.is_cancelled():
            raise asyncio.CancelledError(token.reason)
        return await work
    finally:
        work.cancel()
        stopped.cancel()
        await asyncio.gather(work, stopped, return_exceptions=True)
