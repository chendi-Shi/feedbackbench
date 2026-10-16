"""One preregistered best-effort working-set trim of our own fresh VLA child."""
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run as launcher

stem = ROOT / 'artifacts/frontier/vlajepa_recovery_import_trim_20261016_retry06'
plan_path = stem.with_suffix('.plan.json')
source_path = stem.with_suffix('.source.py')
audit_path = stem.with_suffix('.json')
run_path = ROOT / 'outputs/frontier/vlajepa_recovery/runs/run_0006.json'
protocol_path = ROOT / 'outputs/frontier/vlajepa_recovery/protocol.json'
assert not any(path.exists() for path in (plan_path, source_path, audit_path, run_path))
assert not (run_path.parent.parent / 'active.lock').exists()
assert not (run_path.parent.parent / 'episodes').exists()
utc = lambda: datetime.now(timezone.utc).isoformat()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
source_path.write_bytes(Path(__file__).read_bytes())
plan = {'schema': 'feedbackbench.external_resource_intervention.v1', 'frozen_at': utc(),
    'source_sha256': sha(source_path), 'source': source_path.relative_to(ROOT).as_posix(),
    'recovery_protocol_sha256': sha(protocol_path),
    'recovery_fingerprint': json.loads(protocol_path.read_bytes())['fingerprint'],
    'expected_run': run_path.relative_to(ROOT).as_posix(),
    'trigger_stdout': 'Constructing meta policy and checking the complete checkpoint schema',
    'maximum_api_calls': 1, 'maximum_wall_seconds': 1200,
    'operation': 'K32EmptyWorkingSet only on our newly spawned actual recovery interpreter',
    'identity': 'new descendant, exact script argv, creation time, live process handle and building_meta run status',
    'unchanged': '25 frozen sources, numerical settings, 5.75 GiB preflight, 6 GiB RSS and 0.5 GiB system floor',
    'deviation': 'Additional externally orchestrated resource management; not covered by the original 25-source fingerprint or a matched performance comparison',
    'limitations': ['stdout is not a synchronization barrier; the observed phase before and after the API is recorded.',
        'Working-set pages may return; commit is not released; no guarantee that preflight or recovery succeeds.',
        'No other process will be trimmed or stopped by this controller.'],
    'api_reference': 'https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-emptyworkingset'}
prep_file = ROOT / 'artifacts/frontier/vlajepa_recovery_resource_prep_20261016_retry06.json'
plan['prior_resource_preparation'] = {'path': prep_file.relative_to(ROOT).as_posix(), 'sha256': sha(prep_file)}
plan_path.write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8')
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel.OpenProcess.restype = wintypes.HANDLE
kernel.K32EmptyWorkingSet.argtypes = [wintypes.HANDLE]
kernel.K32EmptyWorkingSet.restype = wintypes.BOOL
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
kernel.GetProcessTimes.restype = wintypes.BOOL
def phase():
    try:
        return json.loads(run_path.read_bytes()).get('status')
    except (FileNotFoundError, json.JSONDecodeError):
        return 'unavailable'

def trim_child(child, spawned_at, event):
    if child.poll() is not None:
        event.update(status='skipped_launcher_exited', api_called=False)
        return
    parent = psutil.Process(child.pid)
    if parent.create_time() != record['launcher_created_at_epoch']:
        event.update(status='skipped_launcher_identity_changed', api_called=False)
        return
    candidates = []
    for process in [parent, *parent.children(recursive=True)]:
        try:
            if (process.name().lower() == 'python.exe' and process.create_time() >= spawned_at - 1
                    and process.cmdline()[-2:] == ['scripts/run_vlajepa_recovery.py', 'run']
                    and process.memory_info().rss >= 128 * 1024**2):
                candidates.append((process, process.create_time()))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    event.update(phase_before=phase(), candidate_count=len(candidates))
    if len(candidates) != 1 or event['phase_before'] != 'building_meta':
        event.update(status='skipped_identity_or_phase_not_confirmed', api_called=False)
        return
    process, created = candidates[0]
    before = process.memory_info()
    event.update(pid=process.pid, created_at_epoch=created, available_bytes_before=psutil.virtual_memory().available,
                 rss_bytes_before=before.rss, private_bytes_before=before.private)
    handle = kernel.OpenProcess(0x1000 | 0x0100, False, process.pid)
    if not handle:
        event.update(status='open_handle_failed', winerror=ctypes.get_last_error(), api_called=False)
        return
    try:
        times = [wintypes.FILETIME() for _ in range(4)]
        assert kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in times))
        handle_created = ((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime) / 1e7 - 11644473600
        assert abs(handle_created-created) < 0.0001
        assert process.create_time() == created and process.cmdline()[-2:] == ['scripts/run_vlajepa_recovery.py', 'run']
        event['phase_at_call'] = phase()
        if event['phase_at_call'] != 'building_meta':
            event.update(status='skipped_phase_changed', api_called=False)
            return
        event['api_started_at'] = utc()
        event['api_attempted'] = True
        audit_path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
        success = bool(kernel.K32EmptyWorkingSet(handle))
        event.update(api_finished_at=utc(), api_called=True, api_success=success,
                     winerror=0 if success else ctypes.get_last_error())
        audit_path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
        after = process.memory_info()
        event.update(status='trim_succeeded' if success else 'trim_failed', phase_after=phase(),
                     rss_bytes_after=after.rss, private_bytes_after=after.private,
                     available_bytes_after=psutil.virtual_memory().available)
    finally:
        kernel.CloseHandle(handle)

env = dict(os.environ)
env['PYTHONUNBUFFERED'] = '1'
env['PYTHONDONTWRITEBYTECODE'] = '1'
env.setdefault('OMP_NUM_THREADS', '2')
env.setdefault('MKL_NUM_THREADS', '2')
spawned_at = time.time()
child = None
record = {'status': 'starting', 'started_at': utc(), 'plan_sha256': sha(plan_path),
          'interventions': [], 'available_bytes_at_launch': psutil.virtual_memory().available}
lines = queue.Queue()
def reader():
    for line in child.stdout:
        lines.put(line)
    lines.put(None)
triggered = False
with (ROOT / '.runtime/vlajepa-recovery-import-trim-20261016-retry06.log').open('x', encoding='utf-8') as log:
    try:
        child = subprocess.Popen([str(launcher.environment_python(ROOT)), '-u', 'scripts/run_vlajepa_recovery.py', 'run'],
            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
        record.update(status='running', launcher_pid=child.pid,
                      launcher_created_at_epoch=psutil.Process(child.pid).create_time())
        threading.Thread(target=reader, daemon=True).start()
        while True:
            if time.time()-spawned_at > plan['maximum_wall_seconds'] and child.poll() is None:
                record['timeout_at'] = utc()
                for owned in psutil.Process(child.pid).children(recursive=True):
                    owned.terminate()
                child.terminate()
                record['status'] = 'own_run_timed_out'
            try:
                line = lines.get(timeout=1)
            except queue.Empty:
                continue
            if line is None:
                break
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
            if not triggered and line.strip() == plan['trigger_stdout']:
                triggered = True
                event = {'signal_observed_at': utc(), 'api_attempted': False}
                record['interventions'].append(event)
                try:
                    trim_child(child, spawned_at, event)
                except Exception as exc:
                    event.update(status='intervention_exception', exception_type=type(exc).__name__, exception_at=utc())
                print(json.dumps({'external_resource_intervention': event}), flush=True)
                audit_path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
        record['runner_exit_code'] = child.wait()
    finally:
        if child is not None and child.poll() is None:
            for owned in psutil.Process(child.pid).children(recursive=True):
                owned.terminate()
            child.terminate()
            child.wait(timeout=30)
        record.update(status='finished' if 'timeout_at' not in record else record['status'],
                      finished_at=utc(), final_run_status=phase(), run_sha256=sha(run_path) if run_path.exists() else None)
        audit_path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'controller_status': record['status'], 'runner_exit_code': record.get('runner_exit_code'),
                  'final_run_status': record['final_run_status']}))
raise SystemExit(record.get('runner_exit_code', 1))
