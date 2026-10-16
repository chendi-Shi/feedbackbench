"""Bounded foreground observation before retrying the frozen recovery runner."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import psutil

root = Path(__file__).resolve().parents[1]
output = root / 'artifacts/frontier/vlajepa_recovery_memory_wait_20261016_02.json'
source = output.with_suffix('.source.py')
assert not output.exists() and not source.exists(), 'Preserve earlier observations'
source.write_bytes(Path(__file__).read_bytes())
started = time.monotonic()
stable_since = None
data = {'status': 'waiting', 'started_at': datetime.now(timezone.utc).isoformat(),
        'minimum_available_gib': 5.75, 'stable_seconds_required': 30,
        'maximum_wait_seconds': 600, 'sampling_seconds': 15, 'samples': [],
        'actions': 'Observe memory only; no services/processes changed, no model invoked',
        'relationship_to_protocol': 'Admission observation only; frozen recovery thresholds unchanged'}
while True:
    now = time.monotonic()
    available = psutil.virtual_memory().available
    if available >= 5.75 * 1024**3:
        stable_since = now if stable_since is None else stable_since
    else:
        stable_since = None
    data['samples'].append({'elapsed_seconds': now-started, 'available_bytes': available})
    if stable_since is not None and now-stable_since >= 30:
        data['status'] = 'stable_memory_gate_passed'
    elif now-started >= 600:
        data['status'] = 'timed_out_below_recovery_gate'
    if data['status'] != 'waiting':
        data['finished_at'] = datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': data['status'], 'elapsed_seconds': round(now-started),
                      'available_gib': round(available/1024**3, 3)}), flush=True)
    if data['status'] != 'waiting':
        raise SystemExit(0 if data['status'] == 'stable_memory_gate_passed' else 3)
    time.sleep(min(15, max(0, 600-(now-started))))
