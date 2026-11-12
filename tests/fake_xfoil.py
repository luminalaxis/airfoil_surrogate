#!/usr/bin/env python3
"""테스트용 가짜 XFOIL: 표준 입력 명령(LOAD, VISC, VPAR N, PACC, ALFA)을 읽어 극선 파일을 쓴다."""
import sys
from pathlib import Path

cmds = sys.stdin.read().splitlines()
coords = Path(cmds[cmds.index("PLOP") + 3].split(maxsplit=1)[1]).read_text().splitlines()
pts = [tuple(map(float, ln.split())) for ln in coords[1:]]
camber = max((u[1] + l[1]) / 2 for u, l in zip(pts, reversed(pts)))  # 대략적인 최대 캠버
re_ = float(next(c for c in cmds if c.startswith("VISC")).split()[1])
polar = cmds[cmds.index("PACC") + 1]
rows = []
for c in cmds:
    if c.startswith("ALFA"):
        a = float(c.split()[1])
        cl = 0.11 * (a + 105 * camber)
        cd = 0.009 * (re_ / 1e5) ** -0.2 + 0.0001 * a**2
        rows.append(f"  {a:7.3f} {cl:8.4f} {cd:9.5f} {cd / 2:9.5f} {-2.5 * camber:8.4f} 0.5000 0.9000")
head = [" XFOIL", "", f" Re = {re_:.3f} e 6", "", "   alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr",
        "  ------ -------- --------- --------- -------- -------- --------"]
Path(polar).write_text("\n".join(head + rows) + "\n")
