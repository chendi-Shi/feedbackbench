"""Authorized one-time desktop page release followed immediately by recovery."""
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
import time
import psutil

root = Path(__file__).resolve().parents[1]
prep = root / 'artifacts/frontier/vlajepa_recovery_resource_prep_20261016_retry06.json'
prep_source = prep.with_suffix('.source.py')
controller = root / '.runtime/run_recovery_with_import_trim_20261016_retry06.py'
assert not any(path.exists() for path in (prep, prep_source, controller))
source = (root / '.runtime/run_recovery_with_import_trim_20261016.py').read_bytes()
for old, new in (
    (b'artifacts/frontier/vlajepa_recovery_import_trim_20261016', b'artifacts/frontier/vlajepa_recovery_import_trim_20261016_retry06'),
    (b'runs/run_0005.json', b'runs/run_0006.json'),
    (b'vlajepa-recovery-import-trim-20261016.log', b'vlajepa-recovery-import-trim-20261016-retry06.log')):
    assert source.count(old) == 1
    source = source.replace(old, new)
anchor = b"plan_path.write_text(json.dumps(plan, indent=2)+'\\n', encoding='utf-8')"
assert source.count(anchor) == 1
source = source.replace(anchor,
    b"prep_file = ROOT / 'artifacts/frontier/vlajepa_recovery_resource_prep_20261016_retry06.json'\n"
    b"plan['prior_resource_preparation'] = {'path': prep_file.relative_to(ROOT).as_posix(), 'sha256': sha(prep_file)}\n" + anchor)
controller.write_bytes(source)
prep_source.write_bytes(Path(__file__).read_bytes())
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel.OpenProcess.restype = wintypes.HANDLE
kernel.K32EmptyWorkingSet.argtypes = [wintypes.HANDLE]
kernel.K32EmptyWorkingSet.restype = wintypes.BOOL
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
kernel.CloseHandle.restype = wintypes.BOOL
record = {'started_at': datetime.now(timezone.utc).isoformat(),
    'source_sha256': hashlib.sha256(prep_source.read_bytes()).hexdigest(),
    'available_bytes_before': psutil.virtual_memory().available, 'targets': [], 'terminated_processes': 0,
    'description': 'One-time working-set release of three identified ChatGPT renderers and OneDrive; no process or service stopped.',
    'limitations': ['Does not deallocate private memory; pages can return. No system paging or experimental limits changed.']}
for process_id, expected in ((11176, 'chatgpt.exe'), (16276, 'chatgpt.exe'), (17204, 'chatgpt.exe'), (14252, 'onedrive.exe')):
    process = psutil.Process(process_id)
    assert process.name().lower() == expected
    if expected == 'chatgpt.exe':
        assert '--type=renderer' in process.cmdline()
    created = process.create_time()
    before = process.memory_info()
    handle = kernel.OpenProcess(0x1000 | 0x0100, False, process.pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        assert process.create_time() == created and process.is_running()
        success = bool(kernel.K32EmptyWorkingSet(handle))
        error = 0 if success else ctypes.get_last_error()
    finally:
        kernel.CloseHandle(handle)
    after = process.memory_info()
    record['targets'].append({'pid': process_id, 'name': expected, 'created_at_epoch': created,
        'api_success': success, 'winerror': error, 'rss_bytes_before': before.rss, 'rss_bytes_after': after.rss})
time.sleep(3)
record.update(finished_at=datetime.now(timezone.utc).isoformat(), available_bytes_after=psutil.virtual_memory().available)
prep.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'resource_preparation': 'completed', 'available_gib': record['available_bytes_after']/1024**3}), flush=True)
runpy.run_path(str(controller), run_name='__main__')
