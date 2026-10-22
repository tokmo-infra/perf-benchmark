from __future__ import annotations

import asyncio
from hashlib import sha256
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Protocol, cast
from urllib.parse import urlparse

import multiprocess
import orjson
import pyarrow as pa
from platformdirs import user_cache_path

from perf_benchmark.pipeline.context import LoaderContext
from perf_benchmark.pipeline.contracts.loader import LoaderPlugin
from perf_benchmark.plugins import register_plugin

_BUILDERS = {"text", "csv", "json", "parquet", "arrow"}
_SUFFIXES = {
    "text": frozenset({".txt"}),
    "csv": frozenset({".csv", ".tsv"}),
    "json": frozenset({".json", ".jsonl", ".ndjson"}),
    "parquet": frozenset({".parquet"}),
    "arrow": frozenset({".arrow"}),
}


@dataclass(frozen=True, slots=True)
class _HFDatasetsSettings:
    dataset: str
    split: str
    data_files: str | tuple[str, ...] | None
    cache_dir: str | None
    batch_rows: int
    load_kwargs: Mapping[str, Any]


def parse_settings(settings: dict[str, Any]) -> _HFDatasetsSettings:
    allowed = {
        "dataset",
        "split",
        "data_files",
        "cache_dir",
        "batch_rows",
        "load_kwargs",
    }
    unknown = sorted(set(settings) - allowed)
    if unknown:
        raise ValueError("unknown hf-datasets settings: " + ", ".join(unknown))
    dataset = settings.get("dataset")
    split = settings.get("split")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("hf-datasets dataset is required")
    if not isinstance(split, str) or not split:
        raise ValueError("hf-datasets split is required")
    load_kwargs = settings.get("load_kwargs", {})
    if not isinstance(load_kwargs, dict):
        raise TypeError("hf-datasets load_kwargs must be an object")
    forbidden = {
        "cache_dir",
        "data_files",
        "keep_in_memory",
        "split",
        "streaming",
    }.intersection(load_kwargs)
    if forbidden:
        raise ValueError(
            "hf-datasets load_kwargs cannot override: " + ", ".join(sorted(forbidden))
        )
    data_files = settings.get("data_files")
    if isinstance(data_files, list):
        data_files = tuple(data_files)
    if data_files is not None and not isinstance(data_files, (str, tuple)):
        raise TypeError("hf-datasets data_files must contain paths")
    if isinstance(data_files, tuple) and not all(
        isinstance(value, str) and value for value in data_files
    ):
        raise TypeError("hf-datasets data_files must contain paths")
    if dataset in _BUILDERS and not data_files:
        raise ValueError("hf-datasets data_files is required for a file builder")
    cache_dir = settings.get("cache_dir")
    if cache_dir is not None and (not isinstance(cache_dir, str) or not cache_dir):
        raise TypeError("hf-datasets cache_dir must be a path")
    batch_rows = settings.get("batch_rows", 128)
    if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows <= 0:
        raise ValueError("hf-datasets batch_rows must be positive")
    return _HFDatasetsSettings(
        dataset,
        split,
        data_files,
        cache_dir,
        batch_rows,
        MappingProxyType(dict(load_kwargs)),
    )


@register_plugin(
    stage="loader",
    name="hf-datasets",
    description="Prepare cached local or Hugging Face datasets and iterate Arrow batches",
    settings_parser=parse_settings,
)
class HFDatasetsLoader(LoaderPlugin):
    def __init__(self, settings: object) -> None:
        if not isinstance(settings, _HFDatasetsSettings):
            raise TypeError("hf-datasets requires parsed settings")
        self._settings = settings

    async def open(self, ctx: LoaderContext) -> _HFDatasetSource:
        settings = self._settings
        data_files = _data_files(settings.dataset, settings.data_files, ctx.base_dir)
        cache_dir = _cache_dir(settings.cache_dir, ctx.base_dir)
        prepared = await asyncio.to_thread(
            _load_dataset,
            _local_dataset_path(settings.dataset, ctx.base_dir),
            data_files,
            settings.split,
            cache_dir,
            settings.load_kwargs,
        )
        return _HFDatasetSource(prepared, settings.batch_rows)


class _HFDatasetSource:
    def __init__(self, dataset, batch_rows: int) -> None:
        self._dataset = dataset.with_format("arrow")
        self._batch_rows = batch_rows
        self._schema = dataset.features.arrow_schema
        self._closed = False

    @property
    def schema(self) -> pa.Schema:
        return self._schema

    def batches(self) -> Iterator[pa.RecordBatch]:
        for value in self._dataset.iter(batch_size=self._batch_rows):
            if self._closed:
                break
            if not isinstance(value, pa.Table):
                raise TypeError("Hugging Face dataset did not produce an Arrow table")
            yield from value.to_batches(max_chunksize=self._batch_rows)

    async def close(self) -> None:
        self._closed = True


def _load_dataset(
    dataset: str,
    data_files: str | tuple[str, ...] | list[str] | None,
    split: str,
    cache_dir: Path,
    load_kwargs: Mapping[str, Any],
):
    import datasets

    datasets.disable_progress_bars()
    cache_dir.mkdir(parents=True, exist_ok=True)
    object_files = _json_object_files(dataset, data_files)
    if object_files is not None:
        from datasets.data_files import DataFilesList

        _use_forkserver()
        files = DataFilesList.from_local_or_remote([str(path) for path in object_files])
        features = _json_object_features(datasets, object_files[0])
        loaded = datasets.Dataset.from_generator(
            _generate_json_objects,
            features=features,
            gen_kwargs={"files": files, "columns": tuple(features)},
            cache_dir=str(cache_dir),
            keep_in_memory=False,
            num_proc=min(8, len(files)),
            split=datasets.NamedSplit(split),
        )
    elif data_files is None and Path(dataset).is_dir():
        try:
            loaded = datasets.load_from_disk(dataset, keep_in_memory=False)
        except FileNotFoundError:
            loaded = datasets.load_dataset(
                dataset,
                split=split,
                cache_dir=str(cache_dir),
                keep_in_memory=False,
                **load_kwargs,
            )
    else:
        loaded = datasets.load_dataset(
            dataset,
            data_files=data_files,
            split=split,
            cache_dir=str(cache_dir),
            keep_in_memory=False,
            **load_kwargs,
        )
    if not isinstance(loaded, datasets.Dataset):
        raise TypeError("hf-datasets loader requires one prepared Dataset split")
    if len(loaded) == 0:
        raise ValueError("dataset is empty")
    return loaded


def _json_object_files(dataset: str, data_files: object) -> tuple[Path, ...] | None:
    if dataset != "json" or not isinstance(data_files, list) or not data_files:
        return None
    paths = tuple(Path(value) for value in data_files if not urlparse(str(value)).scheme)
    if len(paths) != len(data_files) or any(path.suffix.lower() != ".json" for path in paths):
        return None
    markers = {_first_non_whitespace(path) for path in paths}
    if markers == {"{"}:
        return paths
    if "{" in markers:
        raise ValueError("local JSON data_files cannot mix object and array documents")
    return None


def _first_non_whitespace(path: Path) -> str:
    with path.open(encoding="utf-8") as stream:
        while True:
            value = stream.read(1)
            if not value:
                raise ValueError(f"JSON file is empty: {path}")
            if not value.isspace():
                return value


def _generate_json_objects(files, columns):
    for path in files:
        record = _read_json_object(Path(path))
        if set(record) != set(columns):
            raise ValueError(f"JSON object files must have identical top-level fields: {path}")
        yield record


class _ProcessContext(Protocol):
    def get_start_method(self, allow_none: bool = False) -> str | None: ...
    def set_start_method(self, method: str, force: bool = False) -> None: ...


def _use_forkserver() -> None:
    # multiprocess publishes these methods dynamically from its default context.
    context = cast(_ProcessContext, multiprocess)
    if context.get_start_method(allow_none=True) != "forkserver":
        context.set_start_method("forkserver", force=True)


def _json_object_features(datasets, path: Path):
    record = _read_json_object(path)
    return datasets.Features(
        {name: _json_feature(datasets, value) for name, value in record.items()}
    )


def _json_feature(datasets, value: object):
    if value is None or isinstance(value, (dict, list)):
        return datasets.Json()
    if isinstance(value, bool):
        return datasets.Value("bool")
    if isinstance(value, int):
        return datasets.Value("int64")
    if isinstance(value, float):
        return datasets.Value("float64")
    if isinstance(value, str):
        return datasets.Value("string")
    raise TypeError(f"unsupported top-level JSON value: {type(value).__name__}")


def _read_json_object(path: Path) -> dict[str, object]:
    record = orjson.loads(path.read_bytes())
    if not isinstance(record, dict):
        raise TypeError(f"JSON object file must contain one object: {path}")
    return record


def _cache_dir(value: object, base_dir: Path) -> Path:
    if value is None:
        return user_cache_path("perf-benchmark") / "datasets"
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _local_dataset_path(dataset: str, base_dir: Path) -> str:
    if dataset in _BUILDERS or urlparse(dataset).scheme:
        return dataset
    path = Path(dataset).expanduser()
    candidate = path if path.is_absolute() else base_dir / path
    return str(candidate.resolve()) if candidate.exists() else dataset


def _data_files(
    dataset: str, value: str | tuple[str, ...] | None, base_dir: Path,
) -> str | tuple[str, ...] | list[str] | None:
    if dataset not in _BUILDERS:
        return value
    if not isinstance(value, (str, list, tuple)):
        raise ValueError("hf-datasets data_files is required for a file builder")
    requested = (value,) if isinstance(value, str) else tuple(value)
    remote = [item for item in requested if isinstance(item, str) and urlparse(item).scheme]
    local = [item for item in requested if isinstance(item, str) and not urlparse(item).scheme]
    if len(remote) + len(local) != len(requested):
        raise TypeError("hf-datasets data_files must contain paths")
    resolved = list(remote)
    if local:
        resolved.extend(
            str(path)
            for path in resolve_local_files(local, base_dir=base_dir, suffixes=_SUFFIXES[dataset])
        )
    return resolved


def source_identity(settings: Mapping[str, Any], base_dir: Path) -> dict[str, object]:
    """Describe only pinned, local bytes or a pinned Hub commit, without reading rows."""
    parsed = parse_settings(dict(settings))
    if set(parsed.load_kwargs) - {"revision"}:
        return {"status": "unknown", "value": None}
    dataset = _local_dataset_path(parsed.dataset, base_dir)
    if parsed.dataset in _BUILDERS:
        files = _data_files(parsed.dataset, parsed.data_files, base_dir)
        if not isinstance(files, list) or any(urlparse(path).scheme for path in files):
            return {"status": "unknown", "value": None}
        paths = tuple(Path(path) for path in files)
        kind = "local_files"
    elif Path(dataset).exists():
        root = Path(dataset)
        paths = (root,) if root.is_file() else tuple(sorted(path for path in root.rglob("*") if path.is_file()))
        kind = "local_dataset"
    else:
        if urlparse(dataset).scheme:
            return {"status": "unknown", "value": None}
        revision = parsed.load_kwargs.get("revision")
        if not isinstance(revision, str) or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
            return {"status": "unknown", "value": None}
        return {"status": "observed", "value": {"kind": "hub_commit", "dataset": dataset,
                "revision": revision, "split": parsed.split}}
    entries = []
    for index, path in enumerate(paths):
        digest = sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        entries.append({"order": index, "name": path.name, "bytes": path.stat().st_size,
                        "sha256": digest.hexdigest()})
    return {"status": "observed", "value": {"kind": kind, "split": parsed.split,
            "files": entries, "loader": {"dataset": parsed.dataset,
                "batch_rows": parsed.batch_rows, "load_kwargs": dict(parsed.load_kwargs)}}}


def _natural_path_key(path: Path) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", path.name)
    )


def resolve_local_files(
    value: str | tuple[str, ...] | list[str],
    *,
    base_dir: Path,
    suffixes: frozenset[str],
) -> tuple[Path, ...]:
    requested = (value,) if isinstance(value, str) else tuple(value)
    files: list[Path] = []
    for item in requested:
        if not isinstance(item, str) or not item:
            raise TypeError("data_files must contain paths")
        path = Path(item).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        path = path.resolve()
        if path.is_dir():
            files.extend(
                candidate
                for candidate in sorted(path.iterdir(), key=_natural_path_key)
                if candidate.is_file() and candidate.suffix.lower() in suffixes
            )
        elif path.is_file() and path.suffix.lower() in suffixes:
            files.append(path)
        else:
            raise ValueError(f"unsupported or missing dataset path: {path}")
    if not files:
        raise ValueError("dataset contains no supported files")
    return tuple(files)


__all__ = ["HFDatasetsLoader", "parse_settings"]
