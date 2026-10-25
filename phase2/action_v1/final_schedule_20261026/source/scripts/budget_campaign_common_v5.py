"""Revision 5 I/O; original frozen v4 evidence helpers remain unchanged."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

from budget_campaign_common import (ROOT, canonical_hash, case_directory,
    deadline_reached, digest, fingerprint, owned_path, read_json, utc_now,
    verify_episode, verify_protocol)

DEFAULT_CAMPAIGN = ROOT / "outputs/learned_compute/v5"
REPLACE_RETRY_DELAYS = (.05, .1, .2, .4, .5)


def write_json(path, value):
    """Keep old JSON intact while boundedly retrying Windows file contention.

    Exhaustion preserves the pending file for diagnosis and raises. Never
    truncate/unlink the destination, retry serialization, or hide disk errors.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(len(REPLACE_RETRY_DELAYS) + 1):
        try:
            os.replace(temporary, path)
            return
        except OSError as error:
            if (getattr(error, "winerror", None) not in (5, 32, 33)
                    or attempt == len(REPLACE_RETRY_DELAYS)):
                raise
            time.sleep(REPLACE_RETRY_DELAYS[attempt])


def seal_episode(directory):
    """Use the same receipt contract with the revision 5 atomic writer."""
    directory = Path(directory)
    if (directory / "receipt.json").exists():
        raise FileExistsError("receipt already exists")
    files = {path.relative_to(directory).as_posix(): {"sha256": digest(path), "bytes": path.stat().st_size}
             for path in sorted(directory.rglob("*")) if path.is_file() and not path.name.endswith(".tmp")}
    required = {"metadata.json", "result.json", "trajectory.npz", "samples.npz"}
    if not required <= files.keys():
        raise ValueError("cannot seal an incomplete episode")
    write_json(directory / "receipt.json", {"schema": "feedbackbench.budget_episode_receipt.v1",
               "status": "complete", "sealed_at": utc_now(), "files": files})
