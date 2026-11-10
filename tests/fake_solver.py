"""테스트용 가짜 솔버: 저널에서 조건을 읽어 Fluent 형식 report-file을 쓴다.

AF= 가 있으면 물체축 힘계수(cx, cy)를, 없으면 cl, cd를 쓴다.
"""
import math
import re
import sys
from pathlib import Path

journal = Path(sys.argv[1]).read_text(encoding="utf-8")
aoa = float(re.search(r"AOA=(\S+)", journal).group(1))
if aoa >= 99:
    print("Error: Divergence detected in AMG solver")
    sys.exit(0)
if aoa <= -99:
    sys.exit(3)

airfoil = re.search(r"AF=(\S+)", journal)
if airfoil:
    mesh = re.search(r"MESH=(\S+)", journal).group(1)
    assert Path(mesh).is_file(), mesh
    # NACA 4자리 최대 캠버(첫 자리)에 비례하는 가상의 양력, 받음각에 비례하는 항력
    cl = 0.1 * aoa + 0.1 * int(airfoil.group(1)[0])
    cd = 0.02 + 0.001 * aoa
    a = math.radians(aoa)
    cx, cy = cd * math.cos(a) - cl * math.sin(a), cl * math.cos(a) + cd * math.sin(a)
    lines = ['"coef-rfile"', '("Iteration" "cx" "cy")']
    lines += [f"{it} {cx:.10e} {cy:.10e}" for it in range(1, 6)]
else:
    lines = ['"coef-rfile"', '"Iteration" "cl" "cd"', '("Iteration" "cl" "cd")']
    for it in range(1, 11):
        lines.append(f"{it} {0.1 * aoa + it * 0.001:.6e} {0.01 + it * 0.0001:.6e}")
Path("coef-rfile.out").write_text("\n".join(lines) + "\n")
print("calculation complete")
