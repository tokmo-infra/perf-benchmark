"""Offline verification and on-demand SQLite recovery from compressed facts."""
from contextlib import closing
import os
from pathlib import Path
from tempfile import mkstemp

import orjson
import pyarrow.parquet as pq

from .archive import (
    DETAIL_TABLES, SUMMARY_TABLES, _sync, _sync_directory,
    archive_path, detail_lease, read_info,
)
from .schema import ResultDatabase, validate_result_database


def verify(location):
    directory = Path(location).resolve()
    if not directory.is_dir():
        directory = archive_path(directory)
    info = read_info(directory)
    with detail_lease(directory.parent / info['source_name']):
        info = read_info(directory, verify=True, verify_details=True)
        files = orjson.loads(info['files_json'])
        return {'run_id': info['run_id'], 'state': 'verified',
                'files': len(files), 'rows': {f['path']: f['rows'] for f in files},
                'archive_bytes': sum((directory / f['path']).stat().st_size for f in files)
                + (directory / 'summary' / 'artifact_info.parquet').stat().st_size}


def restore(location, output):
    directory = Path(location).resolve()
    if not directory.is_dir():
        directory = archive_path(directory)
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    info = read_info(directory)
    with detail_lease(directory.parent / info['source_name']):
        read_info(directory, verify=True, verify_details=True)
        descriptor, name = mkstemp(prefix='.restore-', suffix='.db', dir=output.parent)
        os.close(descriptor)
        temporary = Path(name)
        try:
            with closing(ResultDatabase(temporary).connect()) as db:
                db.execute('PRAGMA foreign_keys=OFF')
                with db:
                    # The complete detail dictionary includes summary definitions.
                    tables = [(t, 'summary') for t in SUMMARY_TABLES if t not in DETAIL_TABLES]
                    tables.extend((t, 'details') for t in DETAIL_TABLES)
                    for table, tier in tables:
                        db.execute(f'DELETE FROM "{table}"')
                        columns = tuple(r[1] for r in db.execute(f'PRAGMA table_info("{table}")'))
                        parquet = pq.ParquetFile(directory / tier / (table + '.parquet'))
                        if tuple(parquet.schema_arrow.names) != columns:
                            raise ValueError('restore schema mismatch: ' + table)
                        sql = f'INSERT INTO "{table}" VALUES (' + ','.join('?' for _ in columns) + ')'
                        for batch in parquet.iter_batches(batch_size=8192):
                            db.executemany(sql, zip(*(c.to_pylist() for c in batch.columns)))
                    if db.execute('PRAGMA foreign_key_check').fetchone():
                        raise ValueError('restore contains broken result references')
                db.execute('PRAGMA journal_mode=DELETE')
            validate_result_database(temporary)
            _sync(temporary)
            # Exclusive publication preserves any concurrently created output.
            os.link(temporary, output)
            _sync_directory(output.parent)
            return output
        finally:
            for suffix in ('', '-wal', '-shm', '-journal'):
                Path(str(temporary) + suffix).unlink(missing_ok=True)
