"""Persistent, content-addressed cache for completed offline generation artifacts."""

from contextlib import contextmanager
from glob import glob
import hashlib
from pathlib import Path
import shutil
import tempfile

from filelock import FileLock
from platformdirs import user_cache_path
import yaml

from .common import fingerprint, local_path, read_yaml, write_yaml


_MARKER = ".generation-cache.yaml"


def cache_root():
    return user_cache_path("perf-benchmark") / "corpus" / "v1"


def _files(path):
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.is_file())
    raise FileNotFoundError(f"generation source does not exist: {path}")


def _source_paths(config, base):
    mode = config.get("mode")
    if mode == "profile":
        return _files(local_path(base, config["profile"]))
    if mode != "seed":
        return []
    settings = config["source"]["loader"]["settings"]
    dataset = settings["dataset"]
    if dataset not in {"text", "csv", "json", "parquet", "arrow"}:
        return _files(local_path(base, dataset))
    result = []

    def add(value):
        if isinstance(value, dict):
            for part in value.values():
                add(part)
        elif isinstance(value, list):
            for part in value:
                add(part)
        else:
            pattern = str(local_path(base, value))
            matches = sorted(glob(pattern, recursive=True))
            if not matches:
                raise FileNotFoundError(f"generation source does not exist: {pattern}")
            for match in matches:
                result.extend(_files(match))

    add(settings["data_files"])
    return sorted(set(result))


def generation_key(config, tokenizer, base):
    sources = []
    for path in _source_paths(config, base):
        stat = path.stat()
        sources.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    return fingerprint({"generator_version": 3, "recipe": config, "tokenizer": tokenizer.fingerprint,
                        "tokenizer_path": str(tokenizer.path), "sources": sources})


def _contents(path):
    result = {}
    for file in _files(path):
        if file.relative_to(path) == Path(_MARKER):
            continue
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result[str(file.relative_to(path))] = digest.hexdigest()
    return result


def mark(path, key):
    write_yaml(path / _MARKER, {"version": 1, "key": key, "files": _contents(path)})


def valid(path, key):
    if not path.is_dir() or path.is_symlink():
        return False
    try:
        marker = read_yaml(path / _MARKER)
        return marker == {"version": 1, "key": key, "files": _contents(path)}
    except (OSError, ValueError, yaml.YAMLError):
        return False


@contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        yield


def materialize(cached, output, key):
    output = Path(output).absolute()
    with locked(output.parent / ("." + output.name)):
        if valid(output, key):
            return
        if output.is_symlink() or (output.exists() and not output.is_dir()):
            raise FileExistsError("output must be a directory or absent")
        if output.exists() and not (output / _MARKER).is_file():
            raise FileExistsError("existing output is not a cached generation artifact")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".dataset-copy-", dir=output.parent) as temporary:
            staged = Path(temporary) / "artifact"
            shutil.copytree(cached, staged)
            if not valid(staged, key):
                raise ValueError("generation cache changed during copy")
            previous = Path(temporary) / "previous"
            if output.exists():
                output.rename(previous)
            try:
                staged.rename(output)
            except BaseException:
                if previous.exists():
                    previous.rename(output)
                raise


def cache_directory(key):
    return cache_root() / key
