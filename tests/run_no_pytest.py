"""Tiny no-network test harness for environments without pytest.

It supports the simple unit tests in this repository that only use the tmp_path
fixture. Normal development and CI should still use pytest.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    failures: list[str] = []
    tests = sorted(Path(__file__).resolve().parent.glob("test_*.py"))
    count = 0
    for path in tests:
        module = _load_module(path)
        for name, func in sorted(vars(module).items()):
            if not name.startswith("test_") or not callable(func):
                continue
            count += 1
            try:
                _run(func)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{path.name}::{name}: {exc}")
    if failures:
        print(f"FAILED {len(failures)} of {count} tests")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"PASSED {count} tests")
    return 0


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(func) -> None:
    sig = inspect.signature(func)
    kwargs = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in sig.parameters:
            if name == "tmp_path":
                kwargs[name] = Path(tmp)
            else:
                raise RuntimeError(f"Unsupported fixture in no-pytest harness: {name}")
        func(**kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
