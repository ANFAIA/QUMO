"""Local job store: one JSON file per submitted job.

Lets `status JOB` / `result JOB` work from a new process (the CLI is
stateless). Location: $QUMO_HARDWARE_HOME/jobs (default ~/.qumo/hardware/jobs),
never inside the repository. Records contain job ids, backend names, shot
counts, layouts and (for local runs) counts. They never contain credentials.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from .base import JobHandle
from .errors import E_UNKNOWN_JOB, HardwareError

_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def store_dir() -> Path:
    root = os.environ.get("QUMO_HARDWARE_HOME")
    base = Path(root).expanduser() if root else Path.home() / ".qumo" / "hardware"
    d = base / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(job_id: str) -> Path:
    return store_dir() / (_SAFE.sub("_", job_id) + ".json")


def save(job: JobHandle) -> Path:
    path = _path(job.job_id)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(job.to_dict(), f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return path


def load(job_id: str) -> JobHandle:
    path = _path(job_id)
    if not path.exists():
        raise HardwareError(E_UNKNOWN_JOB, f"no job '{job_id}' in the local job store ({path.parent})")
    return JobHandle.from_dict(json.loads(path.read_text()))


def list_jobs(limit: int = 50) -> list[JobHandle]:
    files = sorted(store_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in files[:limit]:
        try:
            out.append(JobHandle.from_dict(json.loads(p.read_text())))
        except (ValueError, TypeError):
            continue
    return out
