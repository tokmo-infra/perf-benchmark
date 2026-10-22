"""Offline compaction of derived reports; native facts and plain historical logs stay intact."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile, ZipFile


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def shared_input(path, store, expected_sha=None):
    """Deduplicate immutable input bytes without changing the loader's file path."""
    path, store = Path(path), Path(store)
    if path.is_symlink() or not path.is_file():
        raise ValueError('input must be a regular file')
    sha = digest(path)
    if expected_sha is not None and sha != expected_sha:
        raise ValueError('selected input SHA mismatch')
    store.mkdir(parents=True, exist_ok=True)
    if hasattr(os, 'geteuid') and os.geteuid() == 0:
        owner = store.parent.stat()
        os.chown(store, owner.st_uid, owner.st_gid)
    cached = store / (sha + path.suffix)
    if cached.is_symlink():
        raise ValueError('shared input must be a regular file')
    try:
        os.link(path, cached)
    except FileExistsError:
        if digest(cached) != sha:
            raise ValueError('shared input content mismatch')
    if not os.path.samefile(path, cached):
        with NamedTemporaryFile(dir=path.parent, prefix='.shared-input-', delete=False) as temporary:
            link = Path(temporary.name)
        try:
            link.unlink()
            os.link(cached, link)
            os.replace(link, path)
        finally:
            link.unlink(missing_ok=True)
    return {'sha256': sha, 'shared_path': str(cached), 'bytes': path.stat().st_size}


def compress_metrics(path):
    """Called only after the observer has closed; verify lossless gzip before replacement."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('metrics must be a regular closed file')
    target = path.with_name(path.name + '.gz')
    if target.exists():
        raise FileExistsError(target)
    before = digest(path)
    with NamedTemporaryFile(dir=path.parent, prefix='.metrics-', delete=False) as temporary:
        pending = Path(temporary.name)
    try:
        with path.open('rb') as source, gzip.open(pending, 'wb', compresslevel=6) as output:
            shutil.copyfileobj(source, output, 1024 * 1024)
        check = hashlib.sha256()
        with gzip.open(pending, 'rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                check.update(block)
        if check.hexdigest() != before or digest(path) != before:
            raise ValueError('metrics changed or gzip verification failed')
        owner = path.stat()
        os.chmod(pending, owner.st_mode & 0o777)
        if hasattr(os, 'geteuid') and os.geteuid() == 0:
            os.chown(pending, owner.st_uid, owner.st_gid)
        os.replace(pending, target)
        path.unlink()
    finally:
        pending.unlink(missing_ok=True)
    return {'path': target.name, 'uncompressed_sha256': before, 'bytes': target.stat().st_size}


def _write(path, value):
    owner = path.stat() if path.exists() else path.parent.stat()
    with NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                            prefix='.compact-', delete=False) as temporary:
        pending = Path(temporary.name)
        json.dump(value, temporary, ensure_ascii=False, indent=2)
        temporary.write('\n')
    os.chmod(pending, owner.st_mode & 0o777 if path.exists() else 0o644)
    if hasattr(os, 'geteuid') and os.geteuid() == 0:
        os.chown(pending, owner.st_uid, owner.st_gid)
    os.replace(pending, path)


def _active(case, root):
    for parent in (case, *case.parents):
        state_path = parent / 'state.json'
        if state_path.is_file():
            state = json.loads(state_path.read_text())
            current = state.get('current') or {}
            if isinstance(current, dict) and current.get('name') == case.name:
                return True
        if parent == root:
            break
    return False


def compact_case(database, root, *, apply=False, compress=False):
    database, root = Path(database), Path(root)
    case = database.parent
    result = {'case': str(case), 'state': 'protected'}
    if database.is_symlink() or case.is_symlink() or _active(case, root):
        return dict(result, reason='active_or_symlink')
    wal = database.with_name(database.name + '-wal')
    if wal.exists() and wal.stat().st_size:
        return dict(result, reason='nonempty_wal')
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro&immutable=1', uri=True) as db:
        rows = db.execute('SELECT status,finished_at FROM runs').fetchall()
        if len(rows) != 1 or rows[0][0] not in ('succeeded', 'failed', 'cancelled') or not rows[0][1]:
            return dict(result, reason='run_not_finished')
        if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('SQLite integrity failed')
    tables = [case / 'run-tables', case / (database.stem + '-tables'),
              case / 'native-tabular-export-20261014']
    workbooks = [folder / 'results.xlsx' for folder in tables if (folder / 'results.xlsx').is_file()]
    if not workbooks:
        return dict(result, reason='no_retained_excel')
    for workbook in workbooks:
        with ZipFile(workbook) as archive:
            if archive.testzip() is not None or 'xl/workbook.xml' not in archive.namelist():
                raise ValueError('retained Excel failed validation')
    targets = {case / 'report.html'}
    for workbook in workbooks:
        targets.update(workbook.parent / name for name in
                       ('summary.csv', 'minutes.csv', 'requests.csv', 'metrics.csv'))
    targets = sorted(p for p in targets if p.is_file() and not p.is_symlink())
    manifest_path = case / 'artifact-manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest is not None:
        if not manifest.get('sqlite_writer_closed') or not isinstance(manifest.get('files'), dict):
            return dict(result, reason='unsupported_or_open_manifest')
        for name, spec in manifest['files'].items():
            path = case / name
            if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink():
                raise ValueError('invalid manifest path')
            if not path.is_file() or path.stat().st_size != spec['bytes'] or digest(path) != spec['sha256']:
                raise ValueError('manifest mismatch: ' + name)
    removed = [{'path': str(p.relative_to(case)), 'bytes': p.stat().st_size,
                'sha256': digest(p)} for p in targets]
    selected = case / 'selected.jsonl'
    expected = None
    selection_path = case / 'selection-manifest.json'
    if selection_path.is_file():
        expected = json.loads(selection_path.read_text()).get('selected_sha256')
    if selected.is_file() and expected is not None and digest(selected) != expected:
        raise ValueError('selected input SHA mismatch')
    result.update(state='planned', remove=removed, historical_logs='plain',
                  native_facts='unchanged', retained_excel=[str(p) for p in workbooks])
    if not apply:
        return result
    before = shutil.disk_usage(root).free
    receipt_path = case / 'storage-compaction.json'
    # Persist the exact deletion intent before removing any derived file.
    _write(receipt_path, dict(result, state='applying'))
    if selected.is_file():
        result['shared_input'] = shared_input(selected, root / '.shared-inputs', expected)
    for path in targets:
        path.unlink()
    metrics = case / 'engine-observation.prom.log'
    if compress and metrics.is_file():
        result['compressed_metrics'] = compress_metrics(metrics)
        result['historical_logs'] = 'gzip_requested_explicitly'
    if manifest is not None:
        for entry in removed:
            manifest['files'].pop(entry['path'], None)
        if 'compressed_metrics' in result:
            manifest['files'].pop(metrics.name, None)
            output = case / result['compressed_metrics']['path']
            manifest['files'][output.name] = {'bytes': output.stat().st_size, 'sha256': digest(output)}
        manifest['storage_policy'] = 'native-facts-and-excel-v1'
        _write(manifest_path, manifest)
        receipt = case / 'mac-sync-receipt.json'
        if receipt.is_file():
            sync = json.loads(receipt.read_text())
            sync['prior_artifact_manifest_sha256'] = sync.get('artifact_manifest_sha256')
            sync['artifact_manifest_sha256'] = digest(manifest_path)
            sync['files'] = manifest['files']
            sync['native_files_verified'] = len(manifest['files'])
            sync['storage_compaction_ref'] = receipt_path.name
            _write(receipt, sync)
    result.update(state='compacted', time_utc=datetime.now(timezone.utc).isoformat(),
                  removed_bytes=sum(row['bytes'] for row in removed),
                  filesystem_available_delta_bytes=shutil.disk_usage(root).free-before)
    _write(receipt_path, result)
    return result


def compact(root, *, apply=False, compress=False):
    root = Path(root).resolve(strict=True)
    outcomes = []
    for database in sorted(root.rglob('run.db')):
        if '.shared-inputs' in database.parts or any(p.is_symlink() for p in database.parents if p != root):
            continue
        try:
            outcomes.append(compact_case(database, root, apply=apply, compress=compress))
        except (OSError, ValueError, BadZipFile, sqlite3.Error) as exc:
            outcomes.append({'case': str(database.parent), 'state': 'protected', 'reason': str(exc)})
    return outcomes
