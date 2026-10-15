"""Launch this project's Python through a short Windows path (no registry changes)."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys


def runtime_directory(root: Path) -> Path:
    identity = hashlib.sha256(str(root.resolve()).casefold().encode()).hexdigest()[:12]
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "FeedbackBench" / f"env-{identity}"


def create_junction(link: Path, target: Path):
    def quote(value):
        return "'" + str(value).replace("'", "''") + "'"
    command = f"New-Item -ItemType Junction -Path {quote(link)} -Target {quote(target)} | Out-Null"
    subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command], check=True)


def environment_python(root: Path) -> Path:
    env = root.resolve() / ".venv"
    if sys.platform != "win32":
        return env / "bin/python"
    if env.resolve() == runtime_directory(root).resolve():
        # Preserve the short lexical path: MSIX may redirect LocalAppData to a
        # much longer physical path when Path.resolve() follows reparse points.
        return runtime_directory(root) / "Scripts/python.exe"
    # LIBERO contains long official BDDL filenames. This directory junction
    # only aliases our existing environment; it copies/moves/deletes no files.
    identity = hashlib.sha256(str(root.resolve()).casefold().encode()).hexdigest()[:12]
    alias = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "FeedbackBench" / identity
    alias.parent.mkdir(parents=True, exist_ok=True)
    if not alias.exists():
        if not env.is_dir():
            raise FileNotFoundError("Create the project .venv with scripts/bootstrap.py first")
        create_junction(alias, env)
    if alias.resolve() != env.resolve():
        raise RuntimeError("The existing short environment alias points to a different directory")
    return alias / "Scripts/python.exe"


def main():
    root = Path(__file__).resolve().parents[1]
    python = environment_python(root)
    if sys.argv[1:] == ["--print-python"]:
        print(python)
        return
    child_env = dict(os.environ)
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"  # Avoid slow bytecode writes into the OneDrive source tree.
    child_env.setdefault("OMP_NUM_THREADS", "2")
    child_env.setdefault("MKL_NUM_THREADS", "2")
    raise SystemExit(subprocess.call([str(python), *sys.argv[1:]], cwd=root, env=child_env))


if __name__ == "__main__":
    main()
