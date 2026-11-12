# Fluent 베이스 케이스 만들기 (examples/naca_lhs, 팀 공통)

`afs run`은 케이스마다 **베이스 케이스를 열고 → 형상 메쉬로 바꾸고 → 유속·받음각·유입 난류·기준값을 바꿔 → 계산**합니다.
난류 모델마다 베이스 케이스가 하나씩 필요합니다 (`[flow] base_cases`).

| 파일 | 난류 모델 | 쓰는 경우 |
|---|---|---|
| `base_sst.cas.h5` | SST k-ω (`kOmegaSST`) | 본 데이터, Airfoil2Vec 검증, XFOIL 비교 |
| `base_sst_lm.cas.h5` | Transition SST (`kOmegaSSTLM`) | XFOIL 비교 (난류 모델 선정) |

**한 명이 만들어 4명이 같은 파일을 씁니다.** 재질·리포트·잔차 기준이 다르면 결과를 합칠 수 없습니다.

## 0. 메쉬 하나 만들기

```bash
cd examples/naca_lhs
```

```bash
afs make-cases
```

```bash
afs --case-list validation mesh --only "v_naca0012*"
```

`runs/_meshes/naca0012_pro-light/mesh.msh` (코드 0.1 m, 첫 셀 높이는 Re 10⁵에서 y⁺ ≈ 1)가 생깁니다.
Fluent Launcher: **2D, Double Precision**, Processes 4.

## 1. 메쉬 읽기

`File > Read > Mesh` → 위 파일. `Domain > Mesh > Check`에 ERROR가 없어야 합니다. 경계 이름은 `main`(익형), `inlet`, `top`, `bottom`, `outlet`.

## 2. General / Models / Materials

- Pressure-Based, **Steady**, 2D **Planar**
- Viscous: `base_sst` → **k-omega SST**, `base_sst_lm` → **Transition SST**
- Materials → air: Density **constant 1.225**, Viscosity **constant 1.7894e-05** (`[flow] rho`, `nu`와 같아야 함. `afs check`가 값을 보여 줌)

### 벽 처리 (y⁺ 모드)
Fluent k-ω SST는 벽 처리를 따로 고르지 않습니다. 첫 셀 y⁺에 따라 **저Re(벽까지 해석)와 벽함수를 자동으로 섞는 y⁺-무관(y⁺-insensitive) 처리**를 씁니다. 그래서 OpenFOAM의 `kqRWallFunction`/`omegaWallFunction` ↔ 저Re 전환은 **메쉬의 y⁺로만** 정해집니다.
- production: y⁺ ≈ 1 (`wall_treatment = "low_re"`)
- validation: y⁺ ≈ 50 (`wall_treatment = "wall_function"`, Airfoil2Vec과 같은 벽함수 영역)
- Transition SST는 y⁺ ≈ 1에서만 씁니다 (설정에서 다른 조합은 오류로 막음).

## 3. Named Expressions

콘솔에 입력합니다 (값은 저널이 케이스마다 바꿈).

```
/define/named-expressions/add "speed" definition "14.607 [m/s]" q
```

```
/define/named-expressions/add "alpha" definition "4 [deg]" q
```

```
/define/named-expressions/add "k_in" definition "0.032 [m^2/s^2]" q
```

```
/define/named-expressions/add "omega_in" definition "326.6 [s^-1]" q
```

`base_sst_lm`에만:

```
/define/named-expressions/add "intermittency_in" definition "1" q
```

k, ω는 도구가 `k = 1.5 (U·I)²`, `ω = √k / (Cμ^¼ · l)` (I = 1%, l = 1% c)로 계산해 넣습니다.

## 4. Boundary Conditions

### `inlet` (velocity-inlet)
- Velocity Specification: **Components**, X = `speed * cos(alpha)`, Y = `speed * sin(alpha)` (드롭다운에서 expression)
- Turbulence Specification: **K and Omega**, Turbulent Kinetic Energy = `k_in`, Specific Dissipation Rate = `omega_in`
- (Transition SST) Intermittency = `intermittency_in`

### `outlet` (pressure-outlet)
Gauge Pressure 0. Backflow 난류도 **K and Omega** = `k_in`, `omega_in`.

### `top`, `bottom` (wall)
Specified Shear, Shear X = 0, Y = 0 (미끄럼 벽). 기본 No Slip으로 두면 바깥 경계에 경계층이 자랍니다.

### `main` (wall)
Stationary, No Slip.

## 5. Reference Values

Compute from **inlet**. 저널이 케이스마다 Velocity, Density, Viscosity, **Length = c**, **Area = c × span**, **Depth = span**을 다시 넣습니다 (Re 모드에 따라 코드가 0.1 m ↔ 1.0 m로 바뀜).

## 6. Solution

- Coupled, Pseudo Time, Least Squares Cell Based, Pressure Second Order, 나머지 Second Order Upwind
- Monitors > Residual: Absolute Criteria **모두 1e-6** (Fluent가 일찍 멈추지 않게. 도구의 잔차 기준은 `[convergence] residual_tol = 1e-5`)

## 7. Report Definitions (이름 정확히)

| Name | 종류 | 설정 | Zone |
|---|---|---|---|
| `cx` | Force Coefficient | Force Vector (1, 0) | main |
| `cy` | Force Coefficient | Force Vector (0, 1) | main |
| `cmz` | Moment Coefficient | Moment Center (0, 0), Axis (0, 0, 1) | main |
| `yplus-max` | Surface Report: Facet Maximum | Wall Yplus | main |
| `yplus-avg` | Surface Report: Area-Weighted Average | Wall Yplus | main |

Report File: Name **`coef-rfile`**, 위 5개, File Name `coef-rfile.out`, 매 반복.

**양력·항력 방향을 받음각마다 돌리지 않는 이유**: 물체축 `cx`, `cy`를 저장하고, 도구가 유입 속도와 **같은 α**로
`CL = cy cos α − cx sin α`, `CD = cx cos α + cy sin α`로 회전합니다. 유입 방향만 돌리고 liftDir/dragDir를 빠뜨리는 버그가
구조적으로 생기지 않습니다. 베이스 케이스에 고정 방향 `cl`, `cd` 리포트를 추가해 두면 도구가 회전값과 비교해
다르면 `summary.csv`의 `force_dir_check = mismatch`와 경고를 냅니다.

**CM**: `cmz`는 앞전(원점) 기준이라 코드 길이와 무관합니다. 도구가 `CM(c/4) = 0.25·cy − cmz` (머리 들림 +)로 옮깁니다.

## 8. (선택) 표면 Cp

`[results] cp_stations`를 켜면 저널의 `${cp_export}` 자리에 아래 명령이 들어갑니다.

```
/file/export/ascii "cp.txt" main () yes pressure-coefficient () no
```

Fluent 버전마다 프롬프트가 다를 수 있으니 GUI에서 `File > Export > Solution Data`(ASCII, main, Pressure Coefficient, Node)를 한 번 저널로 녹화해 `[results] cp_export_command`를 맞추세요. 도구가 윗면·아랫면을 나눠 `cp_u_0.100`, `cp_l_0.100` 같은 열로 보간합니다.

## 9. 저장

`File > Write > Case` → `examples/naca_lhs/base_sst.cas.h5`. Viscous만 Transition SST로 바꾸고 `intermittency_in`과 inlet Intermittency를 넣어 `base_sst_lm.cas.h5`로 한 번 더 저장합니다.

## 10. 파일럿과 팀 실행

```bash
afs check --assignee member1
```

```bash
afs run --assignee member1 --limit 2
```

`runs/production_kOmegaSST_light_main/results.csv`에서 CL·CD·CM, `converged`, `yplus_max`(≈ 1), `n_cells`, `solver_version`을 확인한 뒤 전체를 돌립니다. 끝나면 각자의 `results.csv`를 모아 합칩니다.

```bash
afs merge-results results_all.csv member1/results.csv member2/results.csv member3/results.csv member4/results.csv
```
