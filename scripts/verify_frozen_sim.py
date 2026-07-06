#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通过真实 exe 验收 ORT + GPU（DEFECTS_VERIFY=1，不启动 GUI）。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist" / "缺陷分类系统"
EXE = DIST / "缺陷分类系统.exe"


def main() -> int:
    if not EXE.is_file():
        print("ERROR: missing", EXE)
        return 1
    env = os.environ.copy()
    env["DEFECTS_VERIFY"] = "1"
    print("=== verify_frozen_exe ===")
    print("run:", EXE)
    proc = subprocess.run(
        [str(EXE)],
        cwd=str(DIST),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    print("exit:", proc.returncode)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
