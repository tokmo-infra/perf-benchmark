"""The Windows CLI uses signal.signal and restores the caller's handlers."""
import asyncio
import signal
from types import SimpleNamespace

import pytest

from perf_benchmark.app import main, evidence
from perf_benchmark.runtime.cancellation import CancellationToken


@pytest.mark.parametrize('failure', [False, True])
async def test_windows_cli_cancellation_restores_signal_handlers(monkeypatch, tmp_path, failure):
    token = CancellationToken()
    originals = {sig: object() for sig in (signal.SIGINT, signal.SIGTERM)}
    installed = dict(originals)

    def register(sig, handler):
        previous = installed[sig]
        installed[sig] = handler
        return previous

    async def execute(self, config, **kwargs):
        installed[signal.SIGINT](signal.SIGINT, None)
        await asyncio.sleep(0)
        assert kwargs['cancellation'] is token and token.is_cancelled()
        if failure:
            raise RuntimeError('run failed')
        return 'finished'

    monkeypatch.setattr(main, 'sys', SimpleNamespace(platform='win32'))
    monkeypatch.setattr(signal, 'signal', register)
    monkeypatch.setattr(evidence, 'conditions', lambda *_: {})
    monkeypatch.setattr(evidence, 'quality', lambda *_: {'status': 'eligible'})
    monkeypatch.setattr(main.Coordinator, 'execute', execute)
    config = {'case':'signal handler restoration','application': {'service_capacity': {'enabled': False}}}
    if failure:
        with pytest.raises(RuntimeError, match='run failed'):
            await main.execute_config(config, tmp_path, token)
    else:
        assert await main.execute_config(config, tmp_path, token) == 'finished'
    assert installed == originals


@pytest.mark.parametrize('case',[None,'','  ',23,'multi\nline'])
async def test_case_is_required_before_preparation_or_service_probes(case,tmp_path):
    with pytest.raises(ValueError,match='case is required'):
        await main.execute_config({'case':case},tmp_path)
