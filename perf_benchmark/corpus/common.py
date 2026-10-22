from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def local_path(base, value):
    if not isinstance(value, str) or not value or urlsplit(value).scheme:
        raise ValueError("an explicit local path is required")
    return (Path(base) / value).resolve()


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def read_yaml(path):
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("configuration must be a YAML object")
    return value


def write_yaml(path, value):
    Path(path).write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=True))


@contextmanager
def publish_directory(destination):
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("output directory already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The reservation prevents two producers from publishing to the same target.
    reservation = destination.parent / ("." + destination.name + ".publishing")
    with reservation.open("x"):
        try:
            with tempfile.TemporaryDirectory(prefix=".dataset-", dir=destination.parent) as temporary:
                root = Path(temporary)
                yield root
                if destination.exists() or destination.is_symlink():
                    raise FileExistsError("output directory already exists")
                root.rename(destination)
        finally:
            reservation.unlink()


class RowsWriter:
    """Bounded Parquet batches with an explicit, stable schema."""

    def __init__(self, path, schema):
        self.writer = pq.ParquetWriter(path, schema, compression="zstd")
        self.schema = schema
        self.rows = []

    def append(self, row):
        self.rows.append(row)
        if len(self.rows) >= 128:
            self.flush()

    def flush(self):
        if self.rows:
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
            self.rows.clear()

    def close(self):
        try:
            self.flush()
        finally:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
