import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent


def pytest_sessionstart(session):
    # The annotation library isn't in git; this builds it, or does nothing if it's current.
    subprocess.run([ROOT / "scripts" / "build_annotations.sh"], check=True)
