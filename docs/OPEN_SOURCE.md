# 활용한 오픈소스와 외부 자료

기획서 3장 "학습은 오픈소스로만 한다"에 대응하는 목록입니다. 버전은 개발 PC(Python 3.14, 2026-11-15)에서 확인한 값이고, `pyproject.toml`은 버전을 고정하지 않습니다(`gmsh>=4.11`만 하한).

## 파이썬 라이브러리

| 라이브러리 | 확인 버전 | 라이선스 | 설치 | 쓰는 곳 | 용도 |
|---|---|---|---|---|---|
| [NumPy](https://numpy.org) | 2.5.3 | BSD-3-Clause | 기본 | `cst.py`, `geometry.py`, `meshing.py`, `fluent_msh.py`, `convergence.py`, `dataset.py`, `learn.py` 등 | 좌표·CST 최소제곱 피팅, 메쉬 품질 계산, 수렴 판정, 부트스트랩 |
| [Matplotlib](https://matplotlib.org) | 3.11.1 | Matplotlib License (PSF 계열, BSD 호환) | 기본 | `plots.py`, `dataset.py` | CL-α·CD-α 그래프, 수렴 이력, CST 피팅 비교, 경계 사례 그래프 |
| [Gmsh](https://gmsh.info) | 4.15.2 | GPL-2.0-or-later | 기본 | `meshing.py`, `preflight.py` | 형상별 2D 혼합(경계층 사각형 + 삼각형) 메쉬 생성. pip 패키지에 실행 파일 포함 |
| [LightGBM](https://github.com/microsoft/LightGBM) | 4.7.0 | MIT | `[ml]` | `learn.py` | 비교 모델 1: 평균 회귀 + 분위 회귀(5%, 95%) → 90% 예측 구간, 검증 손실 조기 종료 |
| [PyTorch](https://pytorch.org) | 2.15.0 | BSD-3-Clause 계열 | `[ml]` | `learn.py` | 비교 모델 2: MLP(SiLU) 5개 시드 앙상블, 앙상블 표준편차로 불확실성 |
| [pytest](https://pytest.org) | 9.2.0 | MIT | `[dev]` | `tests/` | 가짜 솔버를 쓴 전체 흐름 테스트 |
| [setuptools](https://setuptools.pypa.io) | 84.0.0 | MIT | 빌드 | `pyproject.toml` | 패키지 설치, `afs` 명령 등록 |

그 밖에는 파이썬 표준 라이브러리만 씁니다(`tomllib` 설정 읽기, `argparse` 명령줄, `csv`/`json` 입출력, `subprocess`·`concurrent.futures`·`multiprocessing` Fluent·메쉬 병렬 실행, `urllib`·`zipfile` 좌표 내려받기 등). scikit-learn은 기획서 목록에 있으나 실제로는 쓰지 않습니다(분할·지표·부트스트랩은 NumPy로 직접 구현).

- **Gmsh(GPL) 주의:** 수업 과제처럼 코드를 배포하지 않거나 소스와 함께 공개하면 문제없습니다. 이 도구를 바이너리로 묶어 비공개 배포하려면 GPL 조건을 확인하세요.

## 데이터

| 자료 | 출처 | 쓰는 곳 |
|---|---|---|
| UIUC Airfoil Coordinates Database (M. Selig, UIUC Applied Aerodynamics Group) | <https://m-selig.ae.illinois.edu/ads/coord_database.html> | 형상 60개 `examples/airfoil_dataset/airfoils/*.dat` (`afs fetch-airfoils`, `airfoil_db.py`) |
| NACA 4자리 형상 공식 | Abbott & von Doenhoff, *Theory of Wing Sections* (1959) | 형상 60개를 코드로 생성 (`geometry.py`) |

## 방법 (참고 문헌)

| 방법 | 문헌 | 쓰는 곳 |
|---|---|---|
| CST 형상 매개변수화 | Kulfan, B. M. (2008). "Universal Parametric Geometry Representation Method." *J. Aircraft* 45(1) | `cst.py` |
| 얇은 익형 이론 (기준 방법) | Anderson, *Fundamentals of Aerodynamics*, 4장 | `cst.py`, `learn.py` (α0 계산, CL = 2π(α − α0)) |
| 메쉬 독립성 GCI | Celik, I. B. et al. (2008). "Procedure for Estimation and Reporting of Uncertainty Due to Discretization in CFD Applications." *J. Fluids Eng.* 130(7) | `mesh_study.py` (`afs gci`) |
| k-ω SST 난류 모델 | Menter, F. R. (1994). *AIAA J.* 32(8) | Fluent 베이스 케이스 설정 |

## 오픈소스가 아닌 것

- **ANSYS Fluent 2026 R1** (상용, 학교 라이선스): 정답(CL, CD) 생성용 CFD 해석기. 학습에는 쓰지 않으며, 기획서상 모델이 근사하는 대상입니다.
