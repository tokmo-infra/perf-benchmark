"""Verify the wheel and rebuild the sdist outside the source checkout."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile


def verify():
    root = Path(__file__).resolve().parents[2]
    distribution = Path(sys.argv[1]) if len(sys.argv) > 1 else root / 'dist'
    wheel, = distribution.glob('*.whl')
    source, = distribution.glob('*.tar.gz')
    with tempfile.TemporaryDirectory(prefix='perf-wheel-') as temporary:
        target = Path(temporary)
        subprocess.run(['uv', 'pip', 'install', '--python', sys.executable, '--target', str(target / 'installed'),
                        '--no-deps', str(wheel)], check=True)
        for directory in ('tests', 'examples'):
            shutil.copytree(root / directory, target / directory,
                            ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
        env = {**os.environ, 'PYTHONPATH': str(target / 'installed')}
        subprocess.run([sys.executable, '-c',
                        'from pathlib import Path; import perf_benchmark; from perf_benchmark import _native; '
                        'assert "installed" in Path(perf_benchmark.__file__).parts; assert _native.now() > 0'],
                       cwd=target, env=env, check=True)
        subprocess.run([sys.executable, '-m', 'pytest', 'tests/executor', '-q', '--import-mode=importlib', '-o', 'asyncio_mode=auto'],
                       cwd=target, env=env, check=True)
        offline = {**env, 'PATH': '', 'CARGO_NET_OFFLINE': 'true'}
        subprocess.run([sys.executable, '-c',
                        'import shutil; assert shutil.which("cargo") is None; '
                        'from perf_benchmark import _native; assert _native.now() > 0'],
                       cwd=target, env=offline, check=True)
        subprocess.run([sys.executable, '-m', 'pytest',
                        'tests/executor/test_business_protocols.py',
                        'tests/executor/test_openai_completions.py',
                        'tests/executor/test_rust_openai_chat.py::test_rust_extension_runs_real_http_and_preserves_native_timestamps',
                        '-q', '--import-mode=importlib', '-o', 'asyncio_mode=auto'],
                       cwd=target, env=offline, check=True)
        with tarfile.open(source) as archive:
            archive.extractall(target / 'source', filter='data')
        extracted, = (target / 'source').iterdir()
        subprocess.run(['uv', 'build', '--wheel', '--out-dir', str(target / 'rebuilt')],
                       cwd=extracted, check=True)


if __name__ == '__main__':
    verify()
