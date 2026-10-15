"""Create the isolated CPU runtime; invoke with Python 3.12 and Git installed."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "64c3cc8be364eb62df4f9fdc4bdccb1be9d1118e"


def run(*args):
    subprocess.run([str(arg) for arg in args], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-url", help="Optional PyPI mirror; TLS remains enabled")
    parser.add_argument("--download", action="store_true", help="Also download pinned models and LIBERO assets")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("The tested runtime uses Python 3.12; invoke this script with Python 3.12.")
    upstream = ROOT / "vendor/MINERVA"
    if not upstream.exists():
        upstream.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "https://github.com/k1000dai/MINERVA.git", upstream)
        run("git", "-C", upstream, "checkout", "--detach", COMMIT)
    actual = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if actual != COMMIT:
        raise SystemExit("Existing vendor checkout is not the pinned commit; preserve it and inspect before proceeding.")
    env = ROOT / ".venv"
    python = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    from run import create_junction, environment_python, runtime_directory
    if not python.exists():
        if sys.platform == "win32":
            physical_env = runtime_directory(ROOT)
            if physical_env.exists():
                raise SystemExit("A cached environment already exists; inspect it before attaching or replacing it.")
            run(sys.executable, "-m", "venv", physical_env)
            create_junction(env, physical_env)
        else:
            run(sys.executable, "-m", "venv", env)
    python = environment_python(ROOT)
    pip = [python, "-m", "pip", "install", "--no-compile", "--timeout", "90"]
    if sys.platform == "win32":
        pip.append("--use-deprecated=legacy-certs")  # TLS still verified; avoids this host's truststore fault.
    # CPU wheels are deliberately installed first to avoid CUDA dependencies.
    run(*pip, "--no-deps", "--index-url", "https://download.pytorch.org/whl/cpu",
        "torch==2.7.1+cpu", "torchvision==0.22.1+cpu")
    index = ["--index-url", args.index_url] if args.index_url else []
    requirements = "requirements-lock.txt" if (ROOT / "requirements-lock.txt").exists() else "requirements-runtime.in"
    # The lock enumerates every installed distribution. --no-deps preserves
    # that tested snapshot and the documented Windows EGL/OpenCV substitutions.
    locked = ["--no-deps"] if requirements == "requirements-lock.txt" else []
    run(*pip, *index, *locked, "-r", requirements)
    # The upstream libero extra excludes Windows. Its EGL probe is Linux-only;
    # runtime.py audits the three required Windows compatibility adaptations.
    run(*pip, *index, "--no-deps", "hf-libero==0.1.4", "robosuite==1.4.0", "robomimic==0.2.0")
    run(*pip, "--no-deps", "--no-build-isolation", "-e", ".")
    if sys.platform == "win32":
        run(python, "scripts/repair_top_level_metadata.py", "--apply")
    if args.download:
        run(python, "scripts/download_models.py")
        run(python, "scripts/download_assets.py")
    print("Ready. Run python scripts/run.py -m embodied_research --help, then the simulator and policy validation scripts.")


if __name__ == "__main__":
    main()
