# airfoil_surrogate

**단일 익형 CFD 데이터셋 생성 + CL·CD 회귀 모델.** 익형 형상(NACA 4자리 / UIUC 좌표) × 받음각 × Re 케이스를 만들고, 형상마다 gmsh로 메쉬를 자동 생성해 ANSYS Fluent 배치 해석을 돌린 뒤, 수렴 상태와 함께 양항계수를 정리합니다. 이 데이터로 CST 형상 계수를 입력으로 하는 **CL·CD 회귀 모델(얇은 익형 이론 기준 / LightGBM / MLP 앙상블)을 학습·평가**합니다.

2D 다요소 날개(메인 + 슬롯 플랩) 위치·형상 최적화 도구 [`aero_automation`](https://github.com/luminalaxis/aero_automation)에서 단일 익형 데이터셋·회귀 모델 부분을 떼어 낸 레포입니다. 케이스 생성, 메쉬, Fluent 배치 실행, 결과 정리 코어는 같은 코드에서 출발했고, 플랩·지면 효과·다운포스·저충실도 스크리닝·단계적 실행은 뺐습니다. 메쉬 이름과 지문(fingerprint)은 분리 전과 같아서, 분리 전에 만든 `runs/_meshes/`를 그대로 쓸 수 있습니다.

```
project.toml (익형 × 받음각 × Re)
   │ fetch-airfoils  UIUC 좌표 파일 내려받기
   │ mesh    형상별 gmsh 메쉬 → Fluent .msh (받음각·Re만 다른 케이스는 메쉬 공유, CPU 코어 수만큼 동시에)
   │         직사각형 영역: inlet / top·bottom(미끄럼 벽) / outlet  (원형 farfield도 가능)
   │ check   Fluent 없이 실행 준비 점검: 베이스 케이스, 메쉬, 경계 이름, 저널, 경로, 디스크·시간 추정
   │ run     케이스별 저널 → fluent 2ddp -t4 -hidden -wait -i run.jou   (--part K/N: 팀원 분담)
   │ collect summary.csv (cl, cd, 수렴 상태 conv_state, y+)
   │ dataset CST 계수 + 형상 단위 분할 → runs/dataset/
   │ train   기준 방법 / LightGBM / MLP → runs/ml/<이름>/   (compare, edge-cases)
   ▼
runs/summary.csv, runs/dataset/, runs/ml/
```

## 설치

파이썬 3.11 이상.

```bash
pip install -e .
```

numpy, matplotlib, gmsh가 함께 설치됩니다. gmsh는 pip 패키지에 실행 파일이 포함돼 있어 따로 설치할 필요 없습니다.

회귀 모델 학습(`afs train`)까지 쓰려면 LightGBM과 PyTorch도 설치합니다.

```bash
pip install -e ".[ml]"
```

Airfoil2Vec 검증(`afs validate airfoil2vec`)에는 parquet 읽기와 Hugging Face 다운로드용 패키지가 필요합니다. XFOIL 비교에는 `xfoil` 실행 파일이 PATH에 있거나 `[validation.xfoil] executable`에 경로를 지정해야 합니다.

```bash
pip install -e ".[validation]"
```

명령은 `afs`(긴 이름 `airfoil-surrogate`도 같음)입니다. 터미널에서 실행하면 큰 글씨 로고 배너가 나오고(색 조합은 실행할 때마다 무작위), 화면을 먼저 지웁니다. 배너는 `--no-banner` 또는 `AFS_NO_BANNER=1`, 화면 지우기는 `--no-clear` 또는 `AFS_NO_CLEAR=1`, 색은 `NO_COLOR=1`로 끕니다. 파일·파이프로 출력을 넘기면 배너·색·화면 지우기가 자동으로 빠집니다.

## 팀 파이프라인: NACA 4자리 100개 × α 0·8·15° (`examples/naca4`)

4명이 **설정 파일 하나**(`examples/naca4/project.toml`)로 같은 조건을 돌립니다. 베이스 케이스는 [docs/FLUENT_BASE_CASE_NACA4.md](docs/FLUENT_BASE_CASE_NACA4.md).

| 항목 | 값 |
|---|---|
| 형상 | NACA 4자리 **100개** (m 0~6, p 2~6, t 6~18 범위의 유효 코드 403개 중 시드 고정 무작위, `[sampling] codes`로 직접 지정 가능) |
| 받음각 | **0°, 8°, 15°** (모든 형상 동일) → 300 케이스, 담당자당 형상 25개 × 3 = 75 케이스 |
| 고정 조건 | 난류 모델 **k-ω SST**, 코드 길이 **100 mm**, Re 10⁵ (U = 14.6 m/s) |
| 해석 | 해석툴 **XFOIL** (Ncrit 9) / **ANSYS Fluent** 2D 정상 상태 |
| 학습 | **MLP + LightGBM**, 입력 (m, p, t, α) → CL, log10 CD, 형상 단위 분할 70/15/15 |

```bash
cd examples/naca4
```

```bash
afs make-cases
```

**1. 해석툴(XFOIL)로 해석** — 300 케이스 전체, 0° → 15°를 1° 간격으로 차례로 풀어 수렴을 돕습니다. 수렴 못 한 α는 `converged = False`로 기록.

```bash
afs xfoil
```

ANSYS Fluent 해석 (각자 담당분, 끝나면 합치기):

```bash
afs run --assignee member1
```

```bash
afs merge-results results_all.csv member1/results.csv member2/results.csv member3/results.csv member4/results.csv
```

**2. ANSYS 해석과 비교** — 같은 case_id끼리 ΔCL, ΔCD/CD, ΔCM을 α별·전체로 (MAE, 편향, R²) 정리하고 parity 그래프를 그립니다. 합친 파일을 쓰려면 `--fluent results_all.csv`.

```bash
afs compare-solvers
```

**3. MLP + LightGBM 학습** — 정답은 ANSYS(기본). `[ml] models = ["lgbm", "mlp"]`, `features = "mpt"`.

```bash
afs dataset
```

```bash
afs train
```

XFOIL 정답으로도 학습해 비교하려면 (형상 분할은 같음):

```bash
afs dataset --source xfoil
```

```bash
afs train --source xfoil
```

**4. 새 에어포일 예측** — `afs train`이 `runs/.../ml/<이름>/models/`에 저장한 LightGBM과 MLP를 불러 CL, CD와 90% 예측 구간을 냅니다.

```bash
afs predict --naca 2412 4415
```

```bash
afs predict --mpt 2.3 4.1 11.7 --alpha 0 8 15
```

```bash
afs predict --random 10
```

```bash
afs predict --airfoil my_foil.dat
```

- `--random N`은 학습 범위 안에서 무작위 NACA (m, p, t)를 뽑습니다. 결과는 `predict.csv`(`--out`으로 경로 지정)에 저장됩니다.
- 학습 범위 밖 입력(외삽)과 학습에 없던 받음각은 `note` 열과 화면에 `*`로 표시합니다. **받음각을 0·8·15° 세 개로만 학습하므로 그 사이 각도(예: 4°)의 예측은 신뢰하기 어렵습니다** (LightGBM은 계단 모양, MLP는 세 점 사이를 임의로 이음).
- 좌표 파일은 최대 두께·최대 캠버·캠버 위치로 (m, p, t)를 근사해 넣습니다. NACA 4자리가 아닌 형상은 근사일 뿐입니다. 임의 형상을 제대로 다루려면 `afs train --features cst16`으로 학습한 모델을 `--run cst16_h64x64`로 쓰세요.
- 두 모델의 예측 차이(`cl_model_spread`, `cd_model_spread_rel`)가 크면 그 예측은 믿기 어렵다는 신호입니다.

모드는 설정의 `[flow]`에서 정하고, 명령줄에서 덮어쓸 수 있습니다. 결과는 모드마다 `runs/<re_mode>_<난류 모델>_<메쉬 모드>_<케이스 목록>/`에 따로 쌓이고, 메쉬는 `runs/_meshes/`에서 난류 모델끼리 공유합니다.

| 옵션 | 값 |
|---|---|
| `--re-mode` | `production` (Re 10⁵, c = 0.1 m, U = 14.6 m/s, y⁺ ≈ 1) / `validation` (Re 4.66×10⁶, y⁺ ≈ 50, 뒷전 sharp, Airfoil2Vec 비교용) |
| `--mesh-mode` | `light` (본 데이터, 약 2만 셀) / `fine` (검증, 약 8만 셀). `[mesh.light]`, `[mesh.fine]` |
| `--turb` | `kOmegaSST` (Fluent SST k-ω). 이 예제는 SST 고정이라 `base_cases`에 SST만 있음 (Transition SST를 쓰려면 베이스 케이스를 추가) |
| `--case-list` | `main` (`cases.csv`) / `validation` (`cases_validation.csv`, validation Re 모드의 기본값) |
| `--assignee` | `cases.csv`의 담당자 열로 내 케이스만 (형상 단위로 배정돼 메쉬도 한 사람만 만듦) |

### 명세 항목 → 구현

| § | 항목 | 구현 |
|---|---|---|
| 0 | U, ν, Re 분리 | `[flow]` chord, nu, rho + 모드별 `velocity` 또는 `reynolds`(나머지는 계산). 두 값을 함께 주면 오류 |
| 0 | 하드코딩 제거 | 물리 조건·모드·k/ω·기준값·베이스 케이스가 `[flow]` 하나에서 저널(`${...}`)과 메쉬로 전달. `[constants]`와 겹치면 오류 |
| 1 | 뒷전 sharp/blunt | `naca_te` = sharp(−0.1036) / blunt(−0.1015). validation 기본 sharp |
| 1 | 코사인 분포 | `geometry.naca4_mpt` |
| 1 | (m, p, t) 실수 입력 | 케이스 변수 `m`, `p`, `t` (NACA 자리 단위: 2412 = 2, 4, 12) |
| 1 | p = 0 예외 | m = 0 또는 p = 0 → 대칭익 (캠버 식 계산 안 함). m > 0인데 p = 0이면 명확한 오류 |
| 2 | 케이스 목록 | `afs make-cases`: `naca_codes`(NACA 정수 코드 × 고정 α, 이 예제) / `nested`(형상 LHS × 형상마다 α 층화) / `joint`(4차원 LHS) |
| 2 | 시드 고정 | `[sampling] seed`. 같은 설정이면 같은 파일, 내용이 다른 기존 파일은 `--force` 없이 덮어쓰지 않음 |
| 2 | cases.csv | `case_id, shape_id, m, p, t, alpha, assignee` (+ `cases.meta.json`) |
| 2 | 검증 목록 분리 | `cases_validation.csv`: `[sampling.validation]` 형상 6개 × α {0, 4, 8} |
| 3 | 첫 셀 높이 자동 | Re와 목표 y⁺(`[flow.*] y_plus`)로 계산 |
| 3 | light / fine | `[mesh.light]`, `[mesh.fine]` 프리셋 |
| 3 | 형상당 메쉬 1회 | 메쉬 이름 = `shape_id` + Re·메쉬 모드 꼬리표. α는 유입 방향 |
| 3 | checkMesh 판정 | `mesh_info.json`: 최대 종횡비, 최대·평균 비직교성(OpenFOAM 정의), 최소 직교 품질 → `quality_flag` (`[mesh] max_aspect_ratio` 등). Fluent `/mesh/quality` 출력도 파싱 |
| 3 | 격자 독립성 3~4단계 | `mesh_level` 스윕 + `afs gci` (가장 조밀한 3단계로 GCI, 4단계면 나머지도 `levels` 열에 기록) |
| 4 | 유입 (U cos α, U sin α) | Named Expression `speed`, `alpha` |
| 4 | liftDir/dragDir 회전 | 물체축 `cx`, `cy`를 유입과 같은 α로 회전 (방향 누락 버그가 생길 수 없음). 고정 방향 `cl/cd` 리포트가 있으면 비교해 `force_dir_check = mismatch` |
| 4 | Aref, lRef, U 연동 | 저널이 케이스마다 Area = c × span, Length = c, Depth = span, Velocity, Density, Viscosity 갱신 |
| 4 | k, ω 자동 | `k = 1.5 (U I)²`, `ω = √k / (Cμ^¼ l)`, I = 1%, l = 1% c → `k_in`, `omega_in` |
| 4 | SST-LM | 별도 베이스 케이스 + `intermittency_in`(γ). Re_θt는 Langtry–Menter 상관식으로 계산해 기록 (Fluent는 유입 Tu에서 내부 계산) |
| 4 | 벽 경계조건 전환 | Fluent k-ω SST는 y⁺-무관 벽 처리라 메쉬 y⁺로 전환. `wall_treatment`와 y⁺·난류 모델 조합을 설정에서 검사 |
| 5 | 담당 케이스만 | `--assignee` |
| 5 | 타임아웃·최대 반복 | `[solver] timeout_minutes`, `[flow] iterations` |
| 5 | 재실행 시 건너뛰기 | 완료(`status.json` done) 케이스 건너뜀 |
| 5 | 로그·소요 시간 | 케이스별 `solver.log`, `transcript.txt`, `status.json` + 결과 폴더의 `run_log.csv` |
| 6 | 마지막 N회 평균 | CL, CD, CM = 마지막 `average_last`회 평균 |
| 6 | 수렴 기준 두 가지 | 잔차 ≤ `residual_tol` (`conv_res_ok`), CL·CD 변동 ≤ 허용치 (`conv_coef_ok`), `criteria = "both"` |
| 6 | converged 플래그 | `converged = False`도 행을 지우지 않고 기록 |
| 6 | y⁺ 최대·평균 | `yplus_max`, `yplus_mean` |
| 6 | Cp 보간 | `[results] cp_stations` → `cp_u_<x/c>`, `cp_l_<x/c>` |
| 7 | 결과 스키마 | `results.csv` 열 고정 (`results.RESULT_COLUMNS`), `afs merge-results`가 열·중복 확인 |
| 8 | Airfoil2Vec | `afs validate airfoil2vec`: metadata 파일만(로컬 또는 HF `hf_hub_download`), inverted −α와 부호 변환 비교, CL ±3%·CD ±10% 판정, light vs fine 오차 |
| 8 | XFOIL | `afs validate xfoil`: subprocess, 같은 Re, Ncrit 9 → SST / SST-LM 비교와 추천 |
| 9 | (m, p, t, α) 입력 | `afs train --features mpt` |
| 9 | 형상 단위 분할 | `[ml] shape_key = "shape_id"` |
| 9 | 미수렴 제외 | `[ml] require_converged = true` |
| 9 | 정규화 | MLP 입력·출력 표준화, 목표 `log10_cd` |

## 예제 구성

인공지능개론 텀프로젝트 기획서("CFD 없이 에어포일 양항계수를 예측하는 회귀 모델")의 데이터 생성부터 모델 평가까지를 담은 예제입니다.

| 항목 | 내용 |
|---|---|
| 형상 | NACA 4자리 60개(공식 생성) + UIUC 60개(`airfoils.csv`). UIUC 중 **캠버 6.5% 이상 10개**(`family = high_camber`: s1223, s1210, e423, ch10sm, fx74cl5140, mh114, dae11, dae21, goe226, e591)는 학습에서 빼고 "처음 보는 형상" 평가에만 씀 |
| 받음각 | −4, −2, 0, 2, 4, 6, 8, 10, 11~16° (14개, 실속 부근 1° 간격) |
| Re | 2×10⁵, 5×10⁵ (코드 0.3 m 기준 유속 9.7, 24.3 m/s) |
| 해석 | Fluent 2D 정상 상태 k-ω SST, 직사각형 영역(앞전 기준 앞 20c, 뒤 40c, 위·아래 20c, 앞 = inlet, 위·아래 = 미끄럼 벽(wall, Specified Shear 0), 뒤 = outlet), 형상당 메쉬 1개(약 5.2만 셀) |
| 입력 → 목표 | CST 계수 16개(면당 8개) + 받음각 + log10(Re) → CL, log10(CD) |
| 라벨 규칙 | 케이스마다 수렴 상태(`conv_state`)와 y+(`yplus_max`, `yplus_avg`) 기록. 수렴 케이스는 마지막 200회, 진동 케이스는 마지막 500회 반복 평균. 발산 케이스는 학습·평가에서 빼고 건수만 보고 |
| 평가 | 조건별 MAE와 표본 수, 형상 단위 부트스트랩 95% 신뢰구간, 90% 예측 구간의 실제 포함률. 실행 간 비교는 공통 케이스만으로(`compare`), 경계 사례 10개(`edge-cases`) |

## 빠른 시작 (예제)

```bash
cd examples/airfoil_dataset
```

```bash
afs fetch-airfoils      # UIUC 좌표 60개를 airfoils/ 에 내려받음 (없는 것만)
```

```bash
afs plan --dry-run      # 3,360 케이스, 형상(메쉬) 120개
```

```bash
afs mesh                # 형상별 메쉬 (각 5~8초)
```

Fluent 베이스 케이스 `base_airfoil.cas.h5`를 만든 뒤([docs/FLUENT_BASE_CASE_AIRFOIL.md](docs/FLUENT_BASE_CASE_AIRFOIL.md)) 팀원 4명이 나눠 실행합니다.

```bash
afs run --part 1/4      # 팀원별로 1/4 ~ 4/4. 형상 30개(840 케이스)씩, 같은 형상은 한 사람에게
```

각자의 `runs/<형상>/` 폴더를 한 `runs/`에 모은 뒤 정리합니다.

```bash
afs collect --no-plots  # summary.csv (cl, cd, 수렴 상태)
```

```bash
afs dataset             # CST 계수 + 형상 단위 분할 → runs/dataset/
```

```bash
afs train               # 기준 방법 / LightGBM / MLP → runs/ml/cst16_h64x64/
```

```bash
afs edge-cases          # 경계 사례 10개 + 사례별 그래프 → runs/ml/cst16_h64x64/edge_cases_mlp/
```

실험 1·2와 14주차 개선 실험은 아래 "학습과 평가"의 표대로 학습한 뒤 `afs compare`로 비교합니다.

- UIUC 사이트에 접속이 안 되면 `coord_seligFmt.zip`(UIUC 전체 묶음)을 받아 `afs fetch-airfoils --zip coord_seligFmt.zip`. 내려받은 `airfoils/*.dat`은 팀원이 같은 좌표를 쓰도록 커밋해 두는 것을 권장합니다.
- `--part K/N`은 `plan`, `mesh`, `run`, `status`, `collect` 등 모든 명령에서 쓸 수 있습니다.

## 수렴 상태 (`conv_state`)

`collect`는 케이스마다 리포트 파일의 반복 이력(마지막 `[convergence] window`개)과 솔버 로그로 수렴 상태를 판정해 `summary.csv`에 넣습니다.

| 상태 | 기준 |
|---|---|
| `converged` | CL 진폭(최대 − 최소) ≤ max(`cl_tol`, `cl_rel_tol`·\|CL\|) 그리고 CD 진폭 ≤ max(`cd_tol`, `cd_rel_tol`·CD), 또는 Fluent가 잔차 기준 만족으로 멈춤 |
| `oscillating` | 그 밖의 경우. `conv_reason`이 `amplitude`(주기적으로 흔들림) 또는 `drift`(아직 한쪽으로 변하는 중) |
| `diverged` | 로그에 `Divergence detected` / `floating point exception`, 값이 NaN·inf, 또는 \|계수\| > `max_abs_coef` |
| `no_data` | 결과 파일 없음 (실행 실패 등) |

함께 기록하는 열: `conv_cl_amp`, `conv_cd_amp`, `conv_cl_std`, `conv_cd_std`(진동 케이스의 정답 불확실성), `conv_cl_drift`, `conv_iters`, `res_continuity` 등 마지막 잔차. `convergence.png` 제목에도 상태가 붙습니다.
`[convergence]` 설정은 모든 프로젝트에서 쓸 수 있습니다(기본: `window = 200`, `cl_tol = 0.005`, `cl_rel_tol = 0.01`, `cd_tol = 1e-4`, `cd_rel_tol = 0.02`, `max_abs_coef = 10`).

- **결과값(정답)**: 마지막 `[results] average_last`(예제: 200)개 반복의 평균입니다. 진동 케이스는 `average_last_oscillating`(예제: 500)개로 더 길게 평균합니다(기획서 라벨 규칙). 실제로 평균에 쓴 반복 수는 `avg_iters` 열에 남습니다.
- **y+**: 베이스 케이스의 리포트 `yplus-max`(벽면 y+ 최대), `yplus-avg`(면적 가중 평균)를 같은 리포트 파일에 넣으면 `summary.csv`의 `yplus-max`, `yplus-avg`와 `dataset.csv`의 `yplus_max`, `yplus_avg`로 기록됩니다([docs/FLUENT_BASE_CASE_AIRFOIL.md](docs/FLUENT_BASE_CASE_AIRFOIL.md) 9단계). `dataset`이 중앙값과 최대를 출력하고, 리포트가 없으면 경고합니다.
- **발산**: `dataset`이 상태별 건수를 출력하고, `train`은 발산 케이스를 학습·평가에서 빼고 분할별 건수만 `metrics.csv`(`state=diverged`)에 남깁니다.

## 데이터셋 (`afs dataset` → `runs/dataset/`)

| 파일 | 내용 |
|---|---|
| `dataset.csv` | 케이스 1건 = 1행. `cst16_u0..u7, cst16_l0..l7`(면당 8개), `cst8_u0..u3, cst8_l0..l3`(면당 4개, 실험 1용), `aoa`, `log10_re`, `cl`, `cd`, `log10_cd`, `conv_state`, `avg_iters`, `yplus_max`, `yplus_avg`, `near_stall`, `aoa_clmax`, `split`, 두께·캠버, 얇은 익형 이론 `alpha0_thin_deg`. Transition SST 결과가 있으면 `cl_tsst`, `log10_cd_tsst`, `d_cl_tsst`, `d_log10_cd_tsst`(T-SST − SST) |
| `airfoils.csv` | 형상별 CST 계수, 피팅 오차(`cst16_rms`, `cst16_max_err`, 코드 기준), 두께·캠버와 위치, 뒷전 두께, 분할 |
| `split.json` | 형상 → `train` / `val` / `test` / `test_ood`. 한 번 만들면 고정(새 형상만 추가 배정). 다시 섞으려면 `dataset --resplit` |
| `cst_fits.png` | 피팅 오차가 가장 큰 형상 6개의 원래 좌표와 CST 재구성 비교 |

- **CST**: 윗면·아랫면을 각각 `y = √x(1−x)·Σ Aᵢ·Bᵢ(x) + x·y_te`(번스타인 다항식)로 최소제곱 피팅합니다. 예제 120개 기준 최대 오차는 NACA 4자리 중앙값 0.0005c(최대 0.002c), UIUC 중앙값 0.001c이고, 좌표 점이 성긴 파일(naca23012, e603)은 앞전 부근에서 0.006c 정도입니다. 최대 오차가 0.005c를 넘으면 경고합니다.
- **분할**: 같은 형상의 모든 받음각·Re 케이스는 같은 분할에 들어갑니다. `holdout_families` 계열은 `test_ood`로만, 나머지는 family(naca4 / uiuc)별로 `split` 비율(70/15/15)에 맞춰 나눕니다. 예제는 학습 77 / 검증 16~17 / 평가 16~17 / 처음 보는 형상 10개입니다.
- **실속 부근**: 형상·Re별로 CL이 최대인 받음각(`aoa_clmax`, 학습에 쓰는 수렴 상태 케이스 기준) ± `near_stall_window`(2°)이면 `near_stall = 1`. 평가에서는 이보다 작은 받음각을 `attached`, 큰 받음각을 `post_stall`로 나눕니다.

## 학습과 평가 (`afs train` → `runs/ml/<이름>/`)

| 모델 | 내용 |
|---|---|
| `baseline` | 얇은 익형 이론 CL = 2π(α − α0), α0는 형상의 캠버선에서 계산. log10 CD = 학습 데이터 평균 |
| `lgbm` | LightGBM. 목표별 평균 모델 + 분위 회귀(5%, 95%) → 90% 예측 구간. 검증 손실로 조기 종료 |
| `mlp` | PyTorch MLP(SiLU), 시드만 다른 5개 앙상블. 평균 = 예측, 앙상블 표준편차 × 1.645 → 90% 예측 구간 |

`metrics.csv`에는 모델 × 분할(`val`, `test`, `test_ood`) × 조건마다 표본 수(`n`, `n_shapes`), `mae_cl`, `mae_log10_cd`, 90% 예측 구간 폭(`width90_*`)과 실제 포함률(`cover90_*`)이 들어가고, MAE와 포함률마다 **형상 단위 부트스트랩 95% 신뢰구간**(`*_lo95`, `*_hi95`)이 붙습니다. 같은 형상의 받음각·Re 케이스는 서로 독립이 아니므로 케이스가 아니라 형상을 복원 추출(2,000회)해 구간을 구합니다.
조건은 `all`(학습에 쓰는 수렴 상태 전체), 유동 구간 `attached`(CL 최대 받음각 − 창보다 작은 받음각, 부착 유동)·`near_stall`(CL 최대 받음각 ± 창)·`post_stall`(+ 창보다 큰 받음각, 실속 후 박리 유동), `state=converged/oscillating`(수렴 상태별)이고, `state=diverged`는 건수(`n`)만 있습니다. 유동 구간은 CFD 정답(CL 최대 받음각)을 알아야 정해지므로 평가 조건으로만 쓰고 모델 입력에는 넣지 않습니다.
`predictions.csv`는 케이스별 정답·예측·구간과 진동 폭, y+, Transition SST 정답(있으면), `run.json`은 설정과 MLP 앙상블 멤버별 조기 종료 지점입니다.
`val`은 조기 종료에 쓰이므로 최종 보고에는 `test`와 `test_ood`를 쓰세요.

기획서의 실험은 옵션으로 학습하고(결과 폴더 이름은 `--name`), `compare`로 같은 케이스끼리 비교합니다.

| 실험 | 학습 | 비교 |
|---|---|---|
| 기본 비교 | `afs train` | `afs compare cst16_h64x64` (기준 방법 대비) |
| 실험 1: 형상 계수 16개 vs 8개 | `afs train --features cst8 --name exp1_cst8` | `afs compare cst16_h64x64 exp1_cst8 --model lgbm mlp` |
| 실험 2: MLP 은닉층 크기 3단계 | `afs train --model mlp --hidden 32,32 --name exp2_small` / `--hidden 128,128,128 --name exp2_large` | `afs compare cst16_h64x64 exp2_small exp2_large --model mlp` |
| 개선 1: 실속 부근 학습 비중 | `afs train --stall-weight 3 --name imp_stall3` | `afs compare cst16_h64x64 imp_stall3 --model lgbm mlp` |
| 개선 2: 처음 보는 계열 일부 학습 | `afs train --ood-train-fraction 0.3 --name imp_ood30` | `afs compare cst16_h64x64 imp_ood30 --model lgbm mlp` |

`[ml]` 설정: `holdout_families`, `split`(3개 비율), `seed`, `near_stall_window`, `train_states`(학습·`all` 조건에 쓰는 수렴 상태, 기본 converged + oscillating), `cst_orders`(면당 계수 수 목록, 기본 `[8, 4]`), `shape_key`(형상 이름 변수, 기본 `airfoil`), `family_key`, `re_key`, `tsst_summary`(Transition SST 결과, 아래 참고).

## 실행 간 비교 (`afs compare` → `runs/ml/_compare/<이름>.csv`)

기획서의 "모든 비교는 동일 분할·동일 지표로"를 지키도록, 여러 학습 결과를 **모든 실행에 공통인 평가 케이스(test, test_ood)만으로** 다시 계산합니다. 개선 2처럼 처음 보는 형상 일부를 학습에 넣은 실행과 비교해도 그 형상은 양쪽에서 모두 빠집니다.

- 기준은 첫 번째 실행의 같은 모델입니다. 실행을 하나만 주면 그 실행의 첫 모델(기본: 기준 방법)이 기준입니다.
- `d_mae_*` = MAE − 기준 MAE(음수 = 개선)와 **짝지은 형상 단위 부트스트랩 95% 신뢰구간**. 모든 모델에 같은 형상 재표집을 써서 차이의 구간을 구하고, 구간이 0을 포함하지 않으면 화면에 `*`를 붙입니다.
- 실행 사이에 `dataset`을 다시 만들어 정답이나 분할이 달라진 케이스는 빼고 경고합니다.

## 경계 사례 (`afs edge-cases` → `runs/ml/<실행>/edge_cases_<모델>/`)

```bash
afs edge-cases --run cst16_h64x64 --model mlp
```

평가 세트(test, test_ood)에서 경계 사례 10개를 골라 `edge_cases.csv`와 사례별 그래프(같은 형상·Re의 CL-α, log10 CD-α: CFD 정답, 모델 예측과 90% 예측 구간, Transition SST 정답)를 만듭니다.

| 범주 | 고르는 케이스 | 개수 |
|---|---|---|
| `stall_pre` | 실속 직전: 받음각 범위 안에서 실속하는 형상·Re의 CL 최대 받음각 이하 (`near_stall`) | 대표 1 + 최악 1 |
| `stall_post` | 실속 직후: CL 최대 받음각 초과 (`near_stall`) | 대표 1 + 최악 1 |
| `turbulence` | SST와 Transition SST의 정답 차이가 큰 케이스 (`tsst_summary` 필요) | 2 |
| `attached_test` | 부착 유동(`attached`), 평가 세트 | 대표 1 + 최악 1 |
| `attached_ood` | 부착 유동, 처음 보는 형상 | 대표 1 + 최악 1 |

- 오차 점수 = \|CL 오차\| / CL MAE + \|log10 CD 오차\| / log10 CD MAE (그 모델의 평가 세트 평균 오차의 몇 배인지). **최악**(실패 사례) = 점수 최대, **대표**(정상 사례) = 점수가 범주 중앙값에 가장 가까운 케이스. 같은 형상은 되도록 한 번만 고릅니다.
- 난류 모델 차이도 같은 척도(모델 평균 오차 대비 배수, `tsst_diff_score`)로 순위를 매깁니다. 평가 분할(val, test, test_ood) 형상만 해당합니다.
- `in_pi90_cl`, `in_pi90_log10_cd`: 정답이 90% 예측 구간 안인지(1/0). 오차가 큰데 구간도 넓으면 모델이 불확실성을 알린 실패, 구간이 좁으면 알리지 못한 실패입니다.

## Transition SST 비교 (신뢰성 검증, 경계 사례)

Re 2×10⁵ 케이스 일부를 Transition SST로 다시 풀고([docs/FLUENT_BASE_CASE_AIRFOIL.md](docs/FLUENT_BASE_CASE_AIRFOIL.md) 맨 아래), `project.toml`에 결과 위치를 적습니다.

```toml
[ml]
tsst_summary = "runs_tsst/summary.csv"
```

`afs dataset`이 이름이 같은 케이스끼리 짝지어 `cl_tsst`, `log10_cd_tsst`, `d_cl_tsst`, `d_log10_cd_tsst`(T-SST − SST)를 붙이고 Re별 차이(중앙값, 최대와 그 케이스)를 출력합니다. 이어서 `train` → `edge-cases`를 다시 실행하면 난류 모델 민감 사례가 채워집니다. Transition SST 정답은 학습에 쓰지 않습니다.

## 실행 옵션

- `--only "naca0012*"`: 일부 케이스만 (와일드카드, 그룹 이름도 가능)
- `--part K/N`: 형상 단위로 N등분한 K번째 몫만 (팀원 분담). `plan`, `mesh`, `run`, `status`, `collect` 등 모든 명령에서 쓸 수 있습니다.
- `-j 2`: 동시 실행 수 (라이선스와 코어 수 고려). `mesh -j 4`는 동시에 만들 메쉬 수(기본: CPU 코어 수, 최대 8. 메쉬당 메모리 약 300 MB)
- `--limit 100`: 남은 케이스 중 앞의 100개만 실행 (나눠서 돌릴 때)
- 중단돼도 다시 `run` 하면 완료된 케이스와 이미 만든 메쉬는 건너뜁니다. 전부 다시 하려면 `run --force`, `mesh --force`.
- 실패 판정: 종료 코드, 로그의 `Error:` / `Divergence detected`, 시간 초과(`timeout_minutes`), 메쉬 생성 실패.
  로그는 `solver.log`(표준 출력)와 `transcript.txt`(저널의 `/file/start-transcript`)를 함께 봅니다. Windows에서 `-hidden`으로 실행하면
  Fluent 출력이 표준 출력으로 나오지 않아서 transcript가 필요합니다. transcript에 다시 찍힌 저널 주석 줄은 검사에서 뺍니다.
- 다시 실행할 때는 그 케이스 폴더의 이전 결과(리포트, 이미지, transcript, `result.cas.h5`/`.dat.h5`)를 먼저 지웁니다.
  Fluent가 덮어쓰기 확인을 물으면 그다음 저널 줄이 답으로 읽혀 배치가 꼬이기 때문입니다.
- `status`는 끝난 케이스의 실행 시간(중앙값)으로 남은 시간을 추정합니다 (`-j`로 동시 실행 수 지정).
- `check`는 Fluent 없이 베이스 케이스, 메쉬(있음·최신·실패, 직교 품질), 세트별 경계 이름, 저널 치환, 리포트 파일 이름, 한글·긴 경로·OneDrive, 디스크와 시간을 점검합니다.

## 메쉬 설정 (`[mesh]`)

| 항목 | 기본값 | 내용 |
|---|---|---|
| `chord` | 1.0 | 코드 [m]. 메쉬 좌표 스케일 (케이스 변수 `chord`가 있으면 그 값) |
| `reynolds`, `y_plus` | 3e6, 1.0 | 첫 셀 높이 계산 (평판 난류 마찰계수 추정). 케이스 변수 `reynolds`가 있으면 형상별 최댓값 |
| `growth_rate` | 1.15 | 경계층 성장률 |
| `bl_thickness` | 0.02 | 경계층 사각형 영역 최대 두께 [c] |
| `surface_cells` | 200 | 윗면·아랫면 각각의 표면 셀 수 (앞전·뒷전에 집중) |
| `te_cells`, `surface_bump` | 6, 0.15 | 뭉툭한 뒷전 면 셀 수, 앞전/뒷전 집중도 |
| `near_size`, `wake_size` | 0.01, 0.01 | 물체 주변 / 후류 셀 크기 [c] |
| `wake_length`, `wake_angles` | 3.0, [-10, 0, 10, 20] | 후류 세분화 길이와 방향 (받음각 범위를 덮도록) |
| `farfield_shape` | `"circle"` | 영역 모양. `"circle"` = 원형 원거리 경계, `"box"` = 직사각형 인클로저 (예제는 `"box"`) |
| `farfield_radius`, `farfield_size` | 50, 5.0 | (circle) 원거리 경계 반지름과 셀 크기 [c] |
| `farfield_split` | false | (circle) true면 원거리 경계를 앞 반원 `inlet`(velocity-inlet) + 뒤 반원 `outlet`(pressure-outlet)으로 나눔. 밀도 일정(비압축) 해석용 |
| `box_upstream`, `box_downstream`, `box_height` | 20, 40, 20 | (box) 앞전 ~ 앞 경계, 앞전 ~ 뒤 경계, 시위선 ~ 위·아래 경계 거리 [c]. 앞 = `inlet`(velocity-inlet), 위·아래 = `top`·`bottom`(wall, Fluent에서 미끄럼 벽 = Specified Shear 0), 뒤 = `outlet`(pressure-outlet) |
| `box_far_size` | 2.0 | (box) 바깥 경계 셀 크기 [c] |
| `min_quality` | 0.05 | 최소 셀 품질 (삼각형 SICN, 사각형 scaled Jacobian). 미달 시 경계층을 얇게 해서 재시도 |
| `timeout` | 300 | 형상 하나의 메쉬 생성 제한 시간 [s]. 넘으면 그 형상만 실패 처리하고 다음 형상으로 넘어감 (0 = 제한 없음) |
| `refine` | 1.0 | 메쉬 전체 세분화 비율 (메쉬 독립성 확인용). 1.5면 모든 셀 크기·첫 셀 높이 ÷1.5, 표면·뒷전 셀 수 ×1.5, 경계층 성장률^(1/1.5) |

경계 이름은 익형 `main`(wall)과 바깥 경계 `farfield`(pressure-far-field, `farfield_split = true`면 `inlet` + `outlet`, `farfield_shape = "box"`면 앞 `inlet` + 위 `top`·아래 `bottom`(wall) + 뒤 `outlet`)입니다. 셀 영역은 `fluid`입니다. 이름이 형상과 관계없이 같아서 Fluent에서 메쉬만 교체해도 설정이 유지됩니다. 메쉬 폴더의 `mesh_domain.png`로 전체 영역 모양을 확인할 수 있습니다.
`[mesh]` 항목과 같은 이름의 케이스 변수가 있으면 케이스 값이 우선합니다. 단 `reynolds`는 형상 안에서 가장 큰 값을 써서 Re가 달라도 메쉬를 공유합니다(예제는 Re 5×10⁵ 기준 첫 셀 높이 → y+ ≤ 1).

**좌표 파일 정리 (자동):** UIUC 같은 좌표 파일은 그대로 쓰면 메쉬가 실패하는 경우가 있어 읽을 때 두 가지를 고칩니다. (1) 뒷전 두 끝점 간격이 0.0005c 미만이면(반올림으로 생긴 틈, 예: S1221) 뾰족한 뒷전으로 닫습니다. (2) 뾰족한 뒷전 바로 앞 윗면·아랫면 점의 쐐기각이 1° 미만이면(칼날형 뒷전, 예: S4180) 그 점 쌍을 지웁니다. 이 처리가 없으면 (1)은 셀 품질 미달, (2)는 gmsh가 멈추는 원인이 됩니다. NACA 4자리 공식 형상은 바뀌지 않습니다.

**동시 생성과 재현성:** `afs mesh`는 CPU 코어 수만큼(기본 최대 8개, 작업 프로세스당 메모리 약 300 MB) 메쉬를 동시에 만듭니다(`-j`로 지정). 예제(4코어)에서 순차 대비 약 3.4배 빠릅니다. 형상마다 새 작업 프로세스에서 시간 제한을 두고 만들기 때문에 한 형상이 멈추거나 실패해도 나머지는 계속 만듭니다. 새 프로세스를 쓰는 이유는 재현성입니다. gmsh는 한 프로세스 안에서 앞서 만든 메쉬에 따라 같은 설정이어도 셀이 조금씩 달라지지만, 새 프로세스의 첫 메쉬는 항상 같습니다. 그래서 만드는 순서, 동시 개수, 팀원 분담(`--part`)과 관계없이 같은 메쉬가 나옵니다.

**품질:** `mesh_info.json`에 gmsh 품질(`min_quality`)과 함께 **Fluent와 같은 정의의 직교 품질**(`min_orthogonal_quality`, `Mesh > Quality`의 Minimum Orthogonal Quality)과 0.1 미만 셀 수를 기록합니다. `check`가 전체 메쉬의 최솟값을 보여주고 0.1 미만이면 경고, 0.01 미만이면 오류로 표시합니다.

### 메쉬 독립성 확인 (`mesh_level`, `refine`, `gci`)

같은 케이스를 메쉬 3단계로 풀어 CL, CD가 메쉬에 따라 얼마나 바뀌는지 보고, 본 배치 메쉬의 이산화 불확실성(GCI)을 구합니다. 순서는 [docs/FLUENT_BASE_CASE_AIRFOIL.md](docs/FLUENT_BASE_CASE_AIRFOIL.md)의 메쉬 독립성 부분을 보세요.

```bash
afs -c project_mesh.toml run
```

```bash
afs -c project_mesh.toml gci
```

- **단계 지정:** `[cases.sweep] mesh_level = [1, 2, 3]`과 `[derived] refine = "1.5 ** (mesh_level - 2)"`. `[mesh]` 항목과 같은 이름의 케이스 변수는 케이스 값이 우선하므로 단계마다 `refine`이 0.67 / 1 / 1.5가 되고, 2단계는 본 배치와 같은 메쉬입니다. 케이스 변수 `mesh_level`이 있으면 메쉬 이름 끝에 `_m<단계>`가 붙어 단계별로 따로 만들어집니다.
- **이름 충돌 방지:** `mesh_level` 없이 케이스마다 `[mesh]` 항목(`refine`, `surface_cells` 등)을 바꾸면, 이름이 같은 메쉬를 공유해 엉뚱한 메쉬로 계산되므로 케이스를 만들 때 오류로 멈춥니다.
- **`gci` 계산 (Celik et al. 2008):** 메쉬 단계만 다르고 나머지 변수가 같은 케이스끼리 묶고, 셀 수로 대표 셀 크기 h = N^(-1/2)를 정한 뒤 CL, CD마다 관측 차수 p, 셀 크기 0으로 외삽한 값, GCI(조밀·보통 메쉬), 수렴 형태(`monotonic` / `oscillatory` / `divergent`)를 `<output_dir>/mesh_study/gci.csv`와 그룹별 `gci_*.png`에 남깁니다.
- **판단 예:** 보통(본 배치) 메쉬의 GCI가 CL 2%, CD 5% 이내면 설정 그대로 진행. 넘으면 `project.toml`의 `[mesh] refine = 1.3`처럼 전체를 조밀하게 하고 다시 확인합니다. 단조 수렴이 아니면 먼저 반복 수렴(`conv_state`)을 확인하세요.

## Fluent 베이스 케이스

해석 설정은 베이스 케이스(`base_airfoil.cas.h5`)에 해 두고, 케이스마다 메쉬만 바꿔 끼웁니다. 화면 단위의 순서, 파일럿 확인, Transition SST 비교 설정은 [docs/FLUENT_BASE_CASE_AIRFOIL.md](docs/FLUENT_BASE_CASE_AIRFOIL.md)를 보세요.

리포트는 물체축 `cx`(Force Vector (1, 0)), `cy`((0, 1))로 만들면 도구가 받음각 θ로 `cl = cy·cosθ − cx·sinθ`, `cd = cx·cosθ + cy·sinθ`를 계산합니다(`[results] wind_axes = true`). 받음각이 바뀌어도 Fluent 쪽 force vector를 바꿀 필요가 없습니다.

저널(`journal.jou`)의 TUI 명령은 Fluent 버전에 따라 프롬프트가 다를 수 있습니다. 오류가 나면 GUI에서 `File > Write > Start Journal`로 같은 작업을 녹화해 해당 줄을 교체하세요.

## 결과물

| 파일 | 내용 |
|---|---|
| `runs/summary.csv` | 케이스별 형상·받음각·Re, 상태, cx/cy, **cl/cd, l/d**, `lift_N_per_m`/`drag_N_per_m`(케이스 변수 `velocity`, `chord`가 있을 때), 반복 수, **수렴 상태 `conv_state`**와 근거(`conv_*`, 마지막 잔차 `res_*`), y+ |
| `runs/ranking.csv` | 형상별 최대 CL과 그때 받음각, 최대 L/D와 그때 받음각·CL·CD (`split_by` 조건별 순위) |
| `runs/plots/*.png` | 형상별 CL-α, CD-α, L/D-α, 극곡선 (`split_by`가 있으면 조건별 폴더) |
| `runs/_meshes/<형상>/` | `geometry.png`, `mesh.png`, `mesh_le.png`, `mesh_domain.png`(전체 영역), `mesh_info.json` |
| `runs/<그룹>/<케이스>/` | `convergence.png`(리포트 수렴 이력), `transcript.txt`(Fluent 출력), `solver.log`(표준 출력), `result.cas.h5` |
| `runs/dataset/` | `dataset.csv`, `airfoils.csv`, `split.json`, `cst_fits.png` (위 "데이터셋") |
| `runs/ml/<이름>/` | `metrics.csv`, `predictions.csv`, `run.json`, `edge_cases_<모델>/` (위 "학습과 평가") |

## 설정 파일 전체 구조 (`project.toml`)

| 섹션 | 내용 |
|---|---|
| `[project]` | `template`(저널), `output_dir`, `max_parallel` |
| `[solver]` | `executable`(비우면 `AWP_ROOTxxx` 환경변수로 자동 탐색), `mode`(2ddp), `processes`, `graphics`(이미지 저장), `timeout_minutes`, `extra_args` |
| `[constants]` | 공통 값 |
| `[cases]` | `csv`(행 단위 케이스, 예제는 `airfoils.csv`), `name`(케이스 폴더 이름 패턴), `group`(분류 폴더 패턴) |
| `[cases.sweep]` | 전체 조합 변수 (예제: `aoa`, `reynolds`) |
| `[[cases.set]]` | 케이스 세트: `label`, `constants`, `sweep`, `name`, `group` |
| `[derived]` | 계산 변수, 예: `velocity = "reynolds * 1.7894e-5 / (1.225 * chord)"` |
| `[mesh]` | 위 메쉬 설정 |
| `[results]` | `report_files`, `average_last`, `average_last_oscillating`(진동 케이스 평균 개수), `x_param`, `wind_axes`, `objective`(`max_ld`/`max_cl`), `split_by`(조건별 분석 변수) |
| `[convergence]` | 수렴 상태 판정 기준 (위 "수렴 상태") |
| `[ml]` | 데이터셋 분할·학습 설정 (위 "학습과 평가") |

케이스 변수 `main_airfoil`이 형상입니다: NACA 4자리 코드(`"0012"`처럼 앞의 0도 유지) 또는 좌표 파일 경로(Selig·Lednicer 형식). 저널 템플릿에서는 모든 케이스 변수에 더해 `${case_name}`, `${group}`, `${case_dir}`, `${project_dir}`, `${nprocs}`, `${mesh_file}`, `${mesh_name}`, `${flow_angle}`(= 받음각)을 쓸 수 있습니다. 포맷 지정도 됩니다: `${aoa:.2f}`.

## 테스트

```bash
pip install -e ".[dev]"
```

```bash
pytest
```

실제 Fluent 대신 가짜 솔버(`tests/fake_solver.py`, `tests/fake_airfoil_solver.py`, 테스트 안의 transcript용 가짜 솔버)로 실행 전 점검(`check`), transcript 오류 검출과 이전 결과 정리, 메쉬 동시 생성과 지문, Fluent 메쉬 면 방향 규칙과 직교 품질, 원형·분할·직사각형 영역 메쉬, 좌표 파일 뒷전 정리, `mesh_level` 메쉬 단계와 GCI, CST 피팅·얇은 익형 이론 α0, 수렴 상태 판정, 형상 단위 분할, 단일 익형 데이터셋 생성부터 세 모델 학습·조건별 평가와 실행 간 비교, Transition SST 비교, 경계 사례 선정까지의 흐름을 검증합니다. 학습 흐름 테스트는 LightGBM과 PyTorch가 설치돼 있을 때만 돌아갑니다.
