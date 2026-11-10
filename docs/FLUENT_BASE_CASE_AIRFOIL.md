# Fluent 베이스 케이스 만들기 (단일 익형 데이터셋, Fluent 2026 R1)

`examples/airfoil_dataset`의 `afs run`은 케이스마다 **베이스 케이스를 열고 → 형상 메쉬로 교체하고 → 유속·받음각·기준값을 바꿔 → 계산**합니다.
물리 모델, 경계조건, 리포트를 담은 `base_airfoil.cas.h5`를 처음 한 번만 만들면 됩니다.
팀원 4명이 **같은 베이스 케이스 파일**을 써야 결과가 일치합니다. 한 명이 만들어 공유하세요.

> 메뉴 이름은 Fluent 버전에 따라 조금씩 다를 수 있습니다. 화면이 다르면 같은 이름의 항목을 Outline View(왼쪽 트리)에서 찾으세요.

## 0. 준비

```bash
cd examples/airfoil_dataset
```

```bash
afs mesh --only "naca0012*"
```

`runs/_meshes/0012_c0.3/mesh.msh`가 생깁니다 (NACA 0012, 코드 0.3 m, 약 5만 셀, 첫 셀 높이는 Re 5×10⁵에서 y+ ≈ 1).
`geometry.png`, `mesh.png`, `mesh_le.png`로 형상과 경계층을, `mesh_domain.png`로 전체 직사각형 영역을 먼저 확인하세요.

Fluent Launcher: Dimension **2D**, **Double Precision**, Solver Processes 4, Working Directory = `examples\airfoil_dataset`

## 1. 메쉬 읽기와 확인

1. `File > Read > Mesh...` → `runs\_meshes\0012_c0.3\mesh.msh`
2. `Domain > Mesh > Check`: 끝에 `Done.`, ERROR 없음
   - Domain Extents가 x −6 ~ 12 m, y −6 ~ 6 m (직사각형: 앞전 기준 앞 20c, 뒤 40c, 위·아래 20c, 코드 0.3 m)
3. `Domain > Mesh > Quality`: Minimum Orthogonal Quality 0.1 이상이면 양호
4. 경계 이름과 타입

| 경계 | 타입 | 위치 |
|---|---|---|
| `inlet` | velocity-inlet | 직사각형의 앞 변 |
| `top` | wall (미끄럼 벽) | 직사각형의 위 변 |
| `bottom` | wall (미끄럼 벽) | 직사각형의 아래 변 |
| `outlet` | pressure-outlet | 직사각형의 뒤 변 |
| `main` | wall | 익형 |
| `interior-fluid` | interior | |

## 2. General

Pressure-Based, **Steady**, 2D Space **Planar**

## 3. Models → Viscous

- **k-omega → SST** (기획서 기준: k-ω SST 완전 난류)
- Transition SST 비교용 베이스 케이스는 맨 아래 "Transition SST 비교"를 보세요.

## 4. Materials → air

Density **constant 1.225**, Viscosity **constant 1.7894e-05** (마하수 0.03~0.07이라 비압축으로 충분)

## 5. Named Expression `speed`, `alpha`

콘솔에 입력합니다.

```
/define/named-expressions/add "speed" definition "9.738 [m/s]" q
```

```
/define/named-expressions/add "alpha" definition "4 [deg]" q
```

확인: `/define/named-expressions/list`에 두 개가 보이면 됩니다.
저널이 케이스마다 `speed`(Re 2×10⁵ → 9.738 m/s, 5×10⁵ → 24.35 m/s)와 `alpha`(받음각)를 바꿉니다.

## 6. Boundary Conditions

### `inlet` (velocity-inlet)
- Velocity Specification Method: **Components**
- X-Velocity: 입력란 오른쪽 드롭다운에서 expression → `speed * cos(alpha)`
- Y-Velocity: expression → `speed * sin(alpha)`
- Turbulence: **Intensity and Viscosity Ratio**, Intensity **0.1 %**, Viscosity Ratio **10**

### `outlet` (pressure-outlet)
- Gauge Pressure 0, Backflow 난류 값은 inlet과 같게

### `top`, `bottom` (wall, 미끄럼 벽)
- 직사각형의 위·아래 변. 두 경계를 함께 선택해 한 번에 설정할 수 있습니다.
- Momentum 탭: Wall Motion **Stationary Wall**, Shear Condition **Specified Shear**, Shear Stress X = **0**, Y = **0** [Pa]
- 기본값(No Slip)으로 두면 바깥 경계를 따라 경계층이 자라서 유동이 막히므로 반드시 바꿉니다.
- 확인: `/define/boundary-conditions/list-zones`에서 `top`, `bottom`이 wall이고, 경계 설정 창의 Shear Condition이 Specified Shear인지 봅니다.
- 위·아래가 벽이라 받음각만큼 기울어진 유입이 위·아래 변에서 수평으로 꺾입니다. 영역 높이(`box_height`, 기본 ±20c)가 충분히 커서 익형 주변의 받음각 영향은 작지만, 파일럿 실행에서 높은 받음각의 cl을 실측값과 비교해 확인하세요.

### `main` (wall)
- Stationary Wall, No Slip (기본값)

## 7. Reference Values

- Compute from: **inlet**
- Area **0.3**, Length **0.3** (코드 0.3 m × 폭 1 m), Density 1.225, Velocity 9.738

케이스마다 저널이 Velocity·Length·Area를 다시 넣습니다.

## 8. Solution Methods / Controls / 잔차 기준

- Pressure-Velocity Coupling **Coupled**, **Pseudo Time Method** 사용
- Gradient Least Squares Cell Based, Pressure **Second Order**, 나머지 **Second Order Upwind**
- `Solution > Monitors > Residual`: **Absolute Criteria를 모두 1e-6**으로 낮춥니다.
  - 기본값(1e-3)이면 힘 계수가 수렴하기 전에 계산이 멈춥니다.
  - 1e-6에 도달하면 Fluent가 "solution is converged"를 출력하고 일찍 끝나며, 도구는 이 경우를 `converged`로 기록합니다.

## 9. Report Definitions와 Report File

`Solution > Report Definitions > New > Force Report > Force...`

| Name | Report Output Type | Force Vector (X, Y) | Zones |
|---|---|---|---|
| `cx` | Force Coefficient | 1, 0 | main |
| `cy` | Force Coefficient | 0, 1 | main |

이름은 정확히 `cx`, `cy`여야 합니다. 도구가 받음각 α로 `cl = cy·cosα − cx·sinα`, `cd = cx·cosα + cy·sinα`를 계산하므로 받음각마다 힘 방향을 바꿀 필요가 없습니다.

기획서의 라벨 규칙대로 케이스마다 y+도 기록합니다. `Solution > Report Definitions > New > Surface Report`

| Name | 종류 | Field Variable | Surfaces |
|---|---|---|---|
| `yplus-max` | Facet Maximum | Turbulence... → Wall Yplus | main |
| `yplus-avg` | Area-Weighted Average | Turbulence... → Wall Yplus | main |

`Solution > Monitors > Report Files > New...`
- Name **`coef-rfile`**, 위 4개 리포트(`cx`, `cy`, `yplus-max`, `yplus-avg`) 추가, File Name `coef-rfile.out`, Get Data Every 1 Iteration

이 파일의 반복별 이력으로 수렴 상태(converged / oscillating / diverged)를 판정하고, 결과값(수렴: 마지막 200회 평균, 진동: 마지막 500회 평균)과 y+(`summary.csv`의 `yplus-max`, `yplus-avg`)를 기록합니다.

## 10. 동작 확인 (권장)

1. `Solution > Initialization`: **Hybrid** → Initialize
2. `Run Calculation`: 20 iterations → Calculate
3. 오류 없이 돌고 `coef-rfile.out`이 생기는지 확인

## 11. 저장

`File > Write > Case...` → `examples\airfoil_dataset\base_airfoil.cas.h5` (Case만 저장, 데이터 불필요)

## 12. 파일럿 실행

```bash
afs run --only "naca0012_Re200000_a4" "naca0012_Re500000_a12"
```

```bash
afs collect --only "naca0012*"
```

확인할 것
- `runs\naca0012\naca0012_Re200000_a4\solver.log`에 Error가 없는지
- `runs\summary.csv`의 해당 행: `cl` ≈ 0.42~0.46, `cd` ≈ 0.011~0.015 (완전 난류 SST, Re 2×10⁵), `conv_state`, `yplus-max`(첫 셀 높이 기준값 1을 크게 넘지 않는지, 예: 5 이하)
- `convergence.png` 제목에 수렴 상태가 표시됨
- 케이스당 소요 시간 (`afs status`) → 3,360 케이스 전체 일정 계산

저널 명령이 버전에 맞지 않으면 GUI에서 `File > Write > Start Journal`로 같은 작업(Replace Mesh, Named Expression 수정, Reference Values 수정)을 녹화해 `journal.jou`의 해당 줄을 교체하세요.

## 메시 독립성과 풍동 실측 비교 (본 생성 전)

기획서의 "해석 신뢰성 확인" 단계입니다. 본 생성과 같은 베이스 케이스로 NACA 0012만 돌립니다.

1. 프로젝트 사본(`project_mesh.toml`, `output_dir = "runs_mesh"`)에서 `[cases.sweep] mesh_level = [1, 2, 3]`을 추가하고
   `[derived] refine = "1.5 ** (mesh_level - 2)"`를 넣습니다. `refine`은 모든 셀 크기·첫 셀 높이를 나누고 표면 셀 수를 곱하는
   `[mesh]` 항목이라 단계마다 메쉬 전체가 일정한 비율로 바뀝니다(2단계 = 본 생성 메쉬). 케이스 이름에 `_m{mesh_level}`을 넣고
   `--only "naca0012*"`로 `mesh`, `run`을 실행합니다. 예: `examples/airfoil_dataset/project_mesh.toml`
2. `afs -c project_mesh.toml gci`로 CL, CD의 Richardson 외삽값과 GCI(이산화 불확실성)를 봅니다.
   보통(2단계) 메쉬의 GCI가 1~2% 이내면 본 생성 설정으로 고정
3. 공개 풍동 실측(예: Re 2×10⁵ 부근 NACA 0012 극선도)과 같은 받음각에서 비교

## Transition SST 비교 (경계 사례용)

1. `base_airfoil.cas.h5`를 열고 Viscous → **Transition SST**, inlet Intermittency 1 → `base_airfoil_tsst.cas.h5`로 저장
2. `project.toml`을 `project_tsst.toml`로 복사해 아래만 바꿉니다.
   ```toml
   [project]
   output_dir = "runs_tsst"
   [constants]
   base_case = "base_airfoil_tsst.cas.h5"
   turbulence_model = "transition-sst"
   ```
3. Re 2×10⁵ 케이스 일부만 실행합니다. 신뢰성 검증용 NACA 0012와, 경계 사례로 쓸 평가(test)·처음 보는 형상(test_ood) 몇 개(예: 3~5개)를 받음각 전체로 돌리는 것을 권장합니다. 형상별 분할은 `runs/dataset/split.json`에서 확인합니다.
   ```bash
   afs -c project_tsst.toml run --only "naca0012_Re200000_*" "s1223_Re200000_*" "e423_Re200000_*"
   ```
4. 본 프로젝트(`project.toml`)의 `[ml]`에 결과 위치를 적고 데이터셋을 다시 만듭니다.
   ```toml
   [ml]
   tsst_summary = "runs_tsst/summary.csv"
   ```
   ```bash
   afs dataset
   ```
   이름이 같은 케이스끼리 짝지어 `dataset.csv`에 `cl_tsst`, `log10_cd_tsst`, `d_cl_tsst`, `d_log10_cd_tsst`(T-SST − SST)가 붙고, Re별 차이(중앙값, 최대와 그 케이스)가 출력됩니다. Re 2×10⁵에서는 층류 박리 기포 때문에 두 모델의 CD 차이가 크게 날 수 있습니다.
5. `afs train` → `afs edge-cases`를 다시 실행하면 경계 사례에 "난류 모델에 따라 정답이 크게 달라지는 케이스" 2개가 들어갑니다(평가 분할 형상만 해당). Transition SST 정답은 학습에 쓰지 않습니다.
