"""Bounded foreground resource observation; never stops processes or runs models."""
import json
from pathlib import Path
import time
from datetime import datetime, timezone
import psutil

output = Path(__file__).with_name('long_retry_memory_wait_20261016.json')
if output.exists():
    raise SystemExit('Do not overwrite a prior wait record')
started = time.monotonic()
stable_since = None
data = {'status': 'waiting', 'started_at': datetime.now(timezone.utc).isoformat(),
        'minimum_available_gib': 6.75, 'stable_seconds_required': 60,
        'maximum_wait_seconds': 600, 'sampling_seconds': 15, 'samples': [],
        'actions': 'Read available memory only; no service/process changes or model invocation'}
while True:
    now = time.monotonic()
    available = psutil.virtual_memory().available
    if available >= 6.75 * 1024**3:
        stable_since = now if stable_since is None else stable_since
    else:
        stable_since = None
    data['samples'].append({'elapsed_seconds': now-started, 'available_bytes': available})
    if stable_since is not None and now-stable_since >= 60:
        data['status'] = 'stable_memory_gate_passed'
    elif now-started >= 600:
        data['status'] = 'timed_out_without_safe_margin'
    output.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': data['status'], 'elapsed_seconds': round(now-started),
                      'available_gib': round(available/1024**3, 3)}), flush=True)
    if data['status'] != 'waiting':
        raise SystemExit(0 if data['status'] == 'stable_memory_gate_passed' else 3)
    time.sleep(15)
