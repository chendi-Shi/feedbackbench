"""One-time authorized working-set release; no termination or memory limits."""
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import psutil

output = Path(__file__).with_name('renderer_memory_release_20261016_03.json')
assert not output.exists()
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel.OpenProcess.restype = wintypes.HANDLE
kernel.K32EmptyWorkingSet.argtypes = [wintypes.HANDLE]
kernel.K32EmptyWorkingSet.restype = wintypes.BOOL
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
kernel.CloseHandle.restype = wintypes.BOOL
targets = []
for process_id in (11176, 16276, 17204):
    process = psutil.Process(process_id)
    assert process.name().lower() == 'chatgpt.exe'
    assert '--type=renderer' in process.cmdline()
    targets.append((process, process.create_time()))
record = {'status': 'releasing_renderer_working_sets',
          'started_at': datetime.now(timezone.utc).isoformat(),
          'available_bytes_before': psutil.virtual_memory().available,
          'targets': [], 'terminated_processes': 0,
          'limitations': ['Working-set release can cause later page faults; it does not deallocate private committed memory.',
                         'No process limits, pagefile, model settings, or frozen experiment thresholds were changed.'],
          'api_reference': 'https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-emptyworkingset'}
for process, created in targets:
    assert process.create_time() == created
    before = process.memory_info()
    handle = kernel.OpenProcess(0x1000 | 0x0100, False, process.pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        success = bool(kernel.K32EmptyWorkingSet(handle))
        error = 0 if success else ctypes.get_last_error()
    finally:
        kernel.CloseHandle(handle)
    after = process.memory_info()
    record['targets'].append({'pid': process.pid, 'created_at_epoch': created,
        'process_name': process.name(), 'renderer_verified': True, 'success': success, 'winerror': error,
        'rss_bytes_before': before.rss, 'rss_bytes_after': after.rss,
        'private_bytes_before': before.private, 'private_bytes_after': after.private})
time.sleep(3)
record.update(status='renderer_working_sets_released',
              available_bytes_after=psutil.virtual_memory().available,
              finished_at=datetime.now(timezone.utc).isoformat())
output.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'status': record['status'], 'successful_releases': sum(x['success'] for x in record['targets']),
                  'available_gib': round(record['available_bytes_after']/1024**3, 3)}))
