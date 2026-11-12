"""테스트용 가짜 Fluent: 저널의 AOA= RE= M= P= T= MESH= CHORD= 를 읽어 Fluent 형식 결과를 쓴다.

- coef-rfile.out: 물체축 cx, cy, 앞전 기준 모멘트 cmz(축 +z), yplus-max, yplus-avg
- 표준 출력: Fluent 버전, 잔차 표, (/mesh/quality 출력)
- cp.txt: 저널에 export/ascii 가 있으면 표면 Cp (x, y, cp)
- M >= 5.5 이면 CL이 진동해 수렴 판정을 못 통과한다.
"""
import math
import re
import sys
from pathlib import Path

journal = Path(sys.argv[1]).read_text(encoding="utf-8")


def val(key):
    return float(re.search(rf"{key}=(\S+)", journal).group(1))


aoa, re_, m, p, t, chord = val("AOA"), val("RE"), val("M"), val("P"), val("T"), val("CHORD")
assert Path(re.search(r"MESH=(\S+)", journal).group(1)).is_file()
alpha0 = -1.05 * m
cl = 0.105 * (aoa - alpha0)
cd = (0.008 + 0.0004 * t + 0.0001 * aoa**2) * (re_ / 1e5) ** -0.2
cm = -0.025 * m                          # c/4 기준, 머리 들림 +
osc = m >= 5.5
a = math.radians(aoa)
lines = ['"coef-rfile"', '("Iteration" "cx" "cy" "cmz" "yplus-max" "yplus-avg")']
n = 400
for it in range(1, n + 1):
    d = math.exp(-it / 30)
    cl_i = cl * (1 + 0.3 * d) + (0.05 * math.sin(it / 5) if osc else 0.0)
    cd_i = cd * (1 + 0.5 * d)
    cx = cd_i * math.cos(a) - cl_i * math.sin(a)
    cy = cl_i * math.cos(a) + cd_i * math.sin(a)
    cmz = 0.25 * cy - cm                 # 앞전 기준, +z(반시계) 부호
    lines.append(f"{it} {cx:.10e} {cy:.10e} {cmz:.10e} {0.8:.4e} {0.35:.4e}")
Path("coef-rfile.out").write_text("\n".join(lines) + "\n")
if "export/ascii" in journal:
    out = ["nodenumber, x-coordinate, y-coordinate, pressure-coefficient"]
    for i in range(201):
        x = 0.5 * (1 - math.cos(math.pi * i / 200))
        yt = 5 * t / 100 * (0.2969 * math.sqrt(x) - 0.126 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        out.append(f"{2 * i + 1}, {x * chord:.8e}, {yt * chord:.8e}, {-1.0 + x:.6e}")    # 윗면 Cp = x − 1
        out.append(f"{2 * i + 2}, {x * chord:.8e}, {-yt * chord:.8e}, {0.5 * x:.6e}")    # 아랫면 Cp = x / 2
    Path("cp.txt").write_text("\n".join(out) + "\n")
print("Welcome to ANSYS Fluent 2026 R1")
print(" Minimum Orthogonal Quality = 2.31e-01 cell 12 on zone 2")
print(" Maximum Aspect Ratio = 5.52e+01 cell 40 on zone 2")
print("  iter  continuity  x-velocity  y-velocity           k       omega     time/iter")
print(f"   {n}  {3e-4 if osc else 4e-6:.4e}  2.1000e-07  3.0000e-07  4.0000e-06  5.0000e-06  0:00:00  0")
print("calculation complete")
