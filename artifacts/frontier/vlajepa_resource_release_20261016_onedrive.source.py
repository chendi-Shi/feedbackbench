"""Release OneDrive's resident pages once while leaving synchronization running."""
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import psutil

output = Path(__file__).with_name('onedrive_memory_release_20261016_03.json')
assert not output.exists()
process = psutil.Process(14252)
assert process.name().lower() == 'onedrive.exe'
created = process.create_time()
before = process.memory_info()
available_before = psutil.virtual_memory().available
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel.OpenProcess.restype = wintypes.HANDLE
kernel.K32EmptyWorkingSet.argtypes = [wintypes.HANDLE]
kernel.K32EmptyWorkingSet.restype = wintypes.BOOL
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
kernel.CloseHandle.restype = wintypes.BOOL
handle = kernel.OpenProcess(0x1000 | 0x0100, False, process.pid)
if not handle:
    raise ctypes.WinError(ctypes.get_last_error())
try:
    assert process.create_time() == created
    success = bool(kernel.K32EmptyWorkingSet(handle))
    error = 0 if success else ctypes.get_last_error()
finally:
    kernel.CloseHandle(handle)
time.sleep(3)
after = process.memory_info()
record = {'status': 'onedrive_working_set_released' if success else 'working_set_release_failed',
          'recorded_at': datetime.now(timezone.utc).isoformat(), 'pid': process.pid,
          'created_at_epoch': created, 'process_name': process.name(), 'success': success, 'winerror': error,
          'rss_bytes_before': before.rss, 'rss_bytes_after': after.rss,
          'private_bytes_before': before.private, 'private_bytes_after': after.private,
          'available_bytes_before': available_before, 'available_bytes_after': psutil.virtual_memory().available,
          'process_still_running': process.is_running(), 'terminated_processes': 0,
          'limitations': ['Pages may become resident again; no private allocation was deallocated.',
                         'No process limits, files, sync configuration, pagefile, or frozen experiment thresholds were changed.'],
          'api_reference': 'https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-emptyworkingset'}
output.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'status': record['status'], 'available_gib': round(record['available_bytes_after']/1024**3, 3)}))
