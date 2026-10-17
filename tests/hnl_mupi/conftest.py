"""Test setup: make ``cafpybara`` importable from its checkout and give core a CAFPYANA_PATH."""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]            # .../cafpybara
sys.path.insert(0, str(REPO.parent))                  # `import cafpybara`
_sibling = REPO.parent / "cafpyana"
if "CAFPYANA_PATH" not in os.environ and _sibling.is_dir():
    os.environ["CAFPYANA_PATH"] = str(_sibling)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: pyhf limit fits (tens of seconds); skipped by the pre-commit hook")
