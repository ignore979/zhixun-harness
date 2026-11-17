from __future__ import annotations

import os
import sys
from importlib.metadata import entry_points
from pathlib import Path


def main() -> int:
    # Redirected Windows output otherwise uses GBK, which rejects SDK emoji logs.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    if len(sys.argv) > 1 and sys.argv[1] in {"swanlab", "trio"}:
        sys.argv = sys.argv[1:]
        entry = next(e for e in entry_points(group="console_scripts") if e.name == sys.argv[0])
        result = entry.load()()
        return result if isinstance(result, int) else 0
    from zhixun.cli import main as application
    return application()


if __name__ == "__main__":
    raise SystemExit(main())
