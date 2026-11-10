"""테스트용 가짜 단일 익형 솔버: 저널의 AOA=, RE=, AF= 를 읽어 반복 이력이 있는 report-file을 쓴다.

- NACA 4자리 코드에서 캠버·두께로 대략적인 양력선과 실속각을 만든다.
- 실속각 + 2° 넘으면 CL이 크게 진동한다 (oscillating 판정 확인용).
- AOA=99 면 Fluent처럼 발산 메시지를 내고 끝난다.
- 6415, Re 2e5, 16°는 계수가 비정상적으로 커져 diverged(coefficient out of range)로 판정된다.
- 결과는 물체축 cx, cy (도구가 받음각으로 회전해 cl, cd 계산)와 벽면 y+ (yplus-max, yplus-avg).
"""
import math
import re
import sys
from pathlib import Path

journal = Path(sys.argv[1]).read_text(encoding="utf-8")
aoa = float(re.search(r"AOA=(\S+)", journal).group(1))
reynolds = float(re.search(r"RE=(\S+)", journal).group(1))
code = re.search(r"AF=(\S+)", journal).group(1)
if aoa >= 99:
    print("Error: Divergence detected in AMG solver: x-momentum")
    sys.exit(0)

m, t = int(code[0]) / 100, int(code[2:]) / 100
alpha0 = -100 * m                                # 캠버 2% → α0 ≈ −2°
stall = 9 + 25 * (t - 0.09) + 2 * math.log10(reynolds / 2e5)
lift_slope = 0.105
cl_stall = lift_slope * (stall - alpha0)
if aoa <= stall:
    cl = lift_slope * (aoa - alpha0)
else:
    cl = cl_stall - 0.06 * (aoa - stall)
cd = 0.007 + 0.3 * t * 0.02 + 0.00015 * aoa**2 + (0.02 * (aoa - stall) if aoa > stall else 0.0)
cd *= (reynolds / 2e5) ** -0.2

blowup = code == "6415" and reynolds < 3e5 and aoa >= 16
yplus = 0.9 * (reynolds / 5e5) ** 0.9 * (1 + 2 * t)   # 첫 셀 높이는 Re 5e5 기준 → y+ ≈ 1

columns = '"Iteration" "cx" "cy" "yplus-max" "yplus-avg"'
lines = ['"coef-rfile"', columns, f"({columns})"]
a = math.radians(aoa)
n = 800
for it in range(1, n + 1):
    decay = math.exp(-it / 40)
    osc = 0.08 * math.sin(it / 6) if aoa > stall + 2 else 0.0
    cl_i = 50.0 if blowup else cl * (1 + 0.3 * decay) + osc
    cd_i = cd * (1 + 0.5 * decay) + 0.1 * abs(osc) * cd
    cx = cd_i * math.cos(a) - cl_i * math.sin(a)
    cy = cl_i * math.cos(a) + cd_i * math.sin(a)
    yp = yplus * (1 + 0.2 * decay)
    lines.append(f"{it} {cx:.10e} {cy:.10e} {yp:.6e} {0.45 * yp:.6e}")
Path("coef-rfile.out").write_text("\n".join(lines) + "\n")
print("  iter  continuity  x-velocity  y-velocity           k       omega     time/iter")
print(f"   {n}  {1e-5 if aoa <= stall else 3e-3:.4e}  2.1000e-07  3.0000e-07  4.0000e-06  5.0000e-06  0:00:00  0")
print("calculation complete")
