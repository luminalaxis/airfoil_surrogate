import csv
import json
import math
import textwrap
from pathlib import Path

import numpy as np
import pytest

from airfoil_surrogate.cases import _coerce, build_cases, partition_cases
from airfoil_surrogate.cli import main
from airfoil_surrogate.config import load_config
from airfoil_surrogate.convergence import ConvergenceConfig, classify, solver_log_info
from airfoil_surrogate.cst import fit_cst, fit_section, section_stats
from airfoil_surrogate.dataset import MLConfig, assign_splits
from airfoil_surrogate.geometry import load_section, naca4, normalize

FAKE_SOLVER = Path(__file__).with_name("fake_airfoil_solver.py")
EXAMPLE = Path(__file__).parents[1] / "examples" / "airfoil_dataset" / "project.toml"


# ---------------------------------------------------------------- CST

def test_cst_symmetric_airfoil():
    fit = fit_section("0012", Path("."), 8)
    assert fit.max_err < 5e-4
    np.testing.assert_allclose(fit.upper, -fit.lower, atol=1e-6)
    assert len(fit.features("cst16")) == 16


def test_cst_reconstructs_cambered_airfoil_and_order_matters():
    xy, le = normalize(naca4("4415"))
    f8, f4 = fit_cst(xy, le, 8), fit_cst(xy, le, 4)
    assert f8.max_err < 1e-3
    assert f8.rms < f4.rms
    yu, yl = f8.evaluate(np.array([0.0, 1.0]))
    assert yu[0] == pytest.approx(0) and yl[0] == pytest.approx(0)
    assert yu[1] - yl[1] == pytest.approx(xy[0, 1] - xy[-1, 1])  # 뭉툭한 뒷전 두께 유지


def test_section_stats_thin_airfoil_theory():
    el = load_section("2412", Path("."), "main")
    s = section_stats(el.xy, el.le_index)
    assert s["alpha0_thin_deg"] == pytest.approx(-2.08, abs=0.1)   # 이론값 −2.08°
    assert s["thickness"] == pytest.approx(0.12, abs=2e-3)
    assert s["camber"] == pytest.approx(0.02, abs=2e-3)  # 같은 x의 윗·아랫면 평균 (NACA는 수직 방향 두께)
    assert s["camber_x"] == pytest.approx(0.4, abs=0.02)


# ---------------------------------------------------------------- 수렴 판정

def _hist(cl, cd=None):
    cl = np.asarray(cl, float)
    return cl, np.full_like(cl, 0.02) if cd is None else np.asarray(cd, float)


def test_convergence_states():
    cfg = ConvergenceConfig(window=100)
    it = np.arange(500)
    steady = _hist(0.8 + 0.5 * np.exp(-it / 20))
    assert classify(steady, cfg)["conv_state"] == "converged"

    osc = classify(_hist(0.8 + 0.05 * np.sin(it / 5)), cfg)
    assert osc["conv_state"] == "oscillating" and osc["conv_reason"].startswith("amplitude")
    assert osc["conv_cl_amp"] == pytest.approx(0.1, rel=0.05)

    drift = classify(_hist(0.5 + 0.001 * it), cfg)
    assert drift["conv_state"] == "oscillating" and drift["conv_reason"].startswith("drift")

    bad = _hist(np.r_[np.full(450, 0.5), np.full(50, np.nan)])
    assert classify(bad, cfg)["conv_state"] == "diverged"
    assert classify(_hist(np.full(500, 50.0)), cfg)["conv_state"] == "diverged"
    assert classify(None, cfg, {"state": "failed", "message": "로그 오류: Divergence detected"})["conv_state"] == "diverged"
    assert classify(None, cfg, {"state": "failed", "message": "종료 코드 3"})["conv_state"] == "no_data"
    # 진동하지만 Fluent가 잔차 기준으로 수렴을 선언한 경우
    assert classify(_hist(0.8 + 0.05 * np.sin(it / 5)), cfg, log_info={"solver_converged": True})["conv_state"] == "converged"


def test_solver_log_residuals(tmp_path):
    log = tmp_path / "solver.log"
    log.write_text(textwrap.dedent("""\
        iter  continuity  x-velocity  y-velocity           k       omega     cx      cy   time/iter
         1  1.0000e+00  1.0e-02  1.0e-02  1.0e-01  1.0e-01  0.01  0.5  0:00:01  2999
      3000  2.5000e-05  3.0e-07  4.0e-07  5.0e-06  6.0e-06  0.02  0.8  0:00:00     0
        """), encoding="utf-8")
    info = solver_log_info(log)
    assert info["res_continuity"] == pytest.approx(2.5e-5)
    assert info["res_omega"] == pytest.approx(6e-6)
    assert "res_cx" not in info and "solver_converged" not in info


# ---------------------------------------------------------------- 분할

def test_shape_level_split_with_holdout_and_stability():
    shapes = {f"naca{i:02d}": "naca4" for i in range(40)}
    shapes |= {f"u{i}": "uiuc" for i in range(20)}
    shapes |= {f"hc{i}": "high_camber" for i in range(5)}
    ml = MLConfig(holdout_families=["high_camber"], seed=1)
    s = assign_splits(shapes, ml)
    assert all(s[f"hc{i}"] == "test_ood" for i in range(5))
    for fam, n in (("naca4", 40), ("uiuc", 20)):
        counts = {k: sum(1 for name, v in s.items() if shapes[name] == fam and v == k) for k in ("train", "val", "test")}
        assert counts == {"train": round(0.7 * n), "val": round(0.15 * n), "test": round(0.15 * n)}
    assert assign_splits(shapes, ml) == s  # 같은 시드면 같은 분할
    # 형상 추가: 기존 배정은 그대로, 새 형상만 배정
    more = shapes | {f"u{i}": "uiuc" for i in range(20, 26)}
    s2 = assign_splits(more, MLConfig(holdout_families=["high_camber"], seed=99), s)
    assert all(s2[k] == v for k, v in s.items())
    assert {s2[f"u{i}"] for i in range(20, 26)} <= {"train", "val", "test"}


def test_csv_keeps_zero_padded_naca_codes():
    assert _coerce("0012") == "0012"
    assert _coerce("2412") == 2412
    assert _coerce("0.5") == 0.5 and _coerce("0") == 0


# ---------------------------------------------------------------- 예제 프로젝트

def test_airfoil_dataset_example_config():
    cfg = load_config(EXAMPLE)
    cases = build_cases(cfg)
    assert len(cases) == 120 * 14 * 2
    assert len({c.mesh.name for c in cases}) == 120          # 형상당 메쉬 1개 (받음각·Re 공유)
    assert all(c.mesh.settings.reynolds == 500000 for c in cases)  # 첫 셀 높이는 큰 Re 기준
    assert all(c.mesh.settings.farfield_shape == "box" for c in cases)
    c = next(c for c in cases if c.name == "naca2412_Re200000_a4")
    assert str(c.params["main_airfoil"]) == "2412" and c.params["flow_angle"] == 4
    assert c.params["velocity"] == pytest.approx(200000 * 1.7894e-5 / (1.225 * 0.3))
    ml = MLConfig.from_dict(cfg.ml)
    fams = {c.params["airfoil"]: c.params["family"] for c in cases}
    assert sum(1 for f in fams.values() if f == "high_camber") == 10
    split = assign_splits(fams, ml)
    assert sum(1 for v in split.values() if v == "train") == 77  # (60 + 50) × 0.7
    # 팀원 4명 분담: 형상 단위로 겹치지 않게 나뉨
    parts = [partition_cases(cases, f"{k}/4") for k in range(1, 5)]
    assert sum(len(p) for p in parts) == len(cases)
    assert [len(p) for p in parts] == [840] * 4
    meshes = [{c.mesh.name for c in p} for p in parts]
    assert all(not (meshes[i] & meshes[j]) for i in range(4) for j in range(i + 1, 4))


def test_split_farfield_mesh(tmp_path):
    from airfoil_surrogate.meshing import make_job, generate_mesh

    job = make_job({"main_airfoil": "0012", "chord": 0.3, "reynolds": 5e5},
                   {"farfield_split": True, "surface_cells": 80, "near_size": 0.03, "wake_size": 0.03,
                    "farfield_radius": 20}, tmp_path)
    info = generate_mesh(job, tmp_path, plots=False)
    assert set(info["zones"]) == {"interior-fluid", "main", "inlet", "outlet"}


def test_box_enclosure_mesh(tmp_path):
    from airfoil_surrogate.fluent_msh import check_msh, read_msh
    from airfoil_surrogate.meshing import make_job, generate_mesh

    job = make_job({"main_airfoil": "2412", "chord": 0.3, "reynolds": 5e5},
                   {"farfield_shape": "box", "box_upstream": 10, "box_downstream": 20, "box_height": 8,
                    "box_far_size": 1.0, "surface_cells": 80, "near_size": 0.03, "wake_size": 0.03}, tmp_path)
    info = generate_mesh(job, tmp_path)
    assert set(info["zones"]) == {"interior-fluid", "main", "inlet", "outlet", "top", "bottom"}
    # 앞 변만 inlet, 위·아래 변은 wall (top/bottom), 뒤 변은 outlet
    msh = read_msh(job.mesh_file)
    nodes = msh["nodes"]
    zone = {name: (zid, bc) for zid, (bc, name) in msh["names"].items()}
    for name, bc, axis, pos in (("inlet", "velocity-inlet", 0, -10), ("top", "wall", 1, 8),
                                ("bottom", "wall", 1, -8), ("outlet", "pressure-outlet", 0, 20)):
        zid, zone_bc = zone[name]
        assert zone_bc == bc
        pts = np.array([nodes[n - 1] for f in msh["zones"][zid] for n in msh["faces"][f][:2]])
        np.testing.assert_allclose(pts[:, axis], pos * 0.3, atol=1e-9)
    check = check_msh(job.mesh_file)
    assert not check["bad_cells"] and not check["open_cells"]
    # 직사각형 영역: x −10c ~ 20c, y ±8c (코드 0.3 m → m 단위)
    np.testing.assert_allclose(nodes.min(0), [-10 * 0.3, -8 * 0.3], atol=1e-9)
    np.testing.assert_allclose(nodes.max(0), [20 * 0.3, 8 * 0.3], atol=1e-9)
    assert (job.dir / "mesh_domain.png").is_file()
    assert info["settings"]["farfield_shape"] == "box"


def test_box_settings_validated(tmp_path):
    from airfoil_surrogate.config import ConfigError
    from airfoil_surrogate.meshing import make_job

    with pytest.raises(ConfigError, match="farfield_shape"):
        make_job({"main_airfoil": "0012"}, {"farfield_shape": "square"}, tmp_path)
    with pytest.raises(ConfigError, match="box_"):
        make_job({"main_airfoil": "0012"}, {"farfield_shape": "box", "box_height": 0}, tmp_path)


# ---------------------------------------------------------------- 전체 흐름

AIRFOILS = [("naca0009", "0009", "naca4"), ("naca0012", "0012", "naca4"), ("naca1412", "1412", "naca4"),
            ("naca2410", "2410", "naca4"), ("naca2412", "2412", "naca4"), ("naca2415", "2415", "naca4"),
            ("naca3412", "3412", "naca4"), ("naca4412", "4412", "naca4"), ("naca4415", "4415", "naca4"),
            ("naca2418", "2418", "naca4"), ("naca6412", "6412", "hc"), ("naca6415", "6415", "hc")]


def make_ml_project(tmp_path):
    (tmp_path / "journal.jou").write_text("AOA=${aoa} RE=${reynolds} AF=${main_airfoil}\n", encoding="utf-8")
    with (tmp_path / "airfoils.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["airfoil", "main_airfoil", "family"])
        w.writerows(AIRFOILS)
    (tmp_path / "project.toml").write_text(textwrap.dedent(f"""
        [project]
        template = "journal.jou"
        max_parallel = 4

        [solver]
        kind = "custom"
        command = ["${{python}}", "{FAKE_SOLVER.as_posix()}", "${{journal}}"]

        [cases]
        csv = "airfoils.csv"
        name = "{{airfoil}}_Re{{reynolds}}_a{{aoa}}"
        group = "{{airfoil}}"

        [cases.sweep]
        aoa = [-2, 0, 2, 4, 6, 8, 10, 12, 13, 14, 15, 16]
        reynolds = [200000, 500000]

        [convergence]
        window = 200

        [results]
        average_last = 200
        average_last_oscillating = 500
        x_param = "aoa"

        [ml]
        holdout_families = ["hc"]
        seed = 3
        tsst_summary = "runs_tsst/summary.csv"
    """), encoding="utf-8")
    return tmp_path / "project.toml"


def test_dataset_and_training_workflow(tmp_path, capsys):
    pytest.importorskip("lightgbm")
    pytest.importorskip("torch")
    cfg_path = make_ml_project(tmp_path)
    assert main(["-c", str(cfg_path), "run"]) == 0
    summary = list(csv.DictReader((tmp_path / "runs" / "summary.csv").open(encoding="utf-8-sig")))
    states = {r["conv_state"] for r in summary}
    assert states == {"converged", "oscillating", "diverged"}
    row = next(r for r in summary if r["case"] == "naca2412_Re200000_a4")
    assert float(row["cl"]) == pytest.approx(0.105 * 6, rel=1e-3)   # cx, cy → 받음각 회전 → cl
    assert row["conv_state"] == "converged" and float(row["res_continuity"]) == pytest.approx(1e-5)
    assert (tmp_path / "runs" / "naca2412" / "naca2412_Re200000_a4" / "convergence.png").is_file()
    # 라벨 규칙: 수렴 케이스는 마지막 200회, 진동 케이스는 마지막 500회 평균. y+도 기록
    osc = next(r for r in summary if r["conv_state"] == "oscillating")
    assert int(row["avg_iters"]) == 200 and int(osc["avg_iters"]) == 500
    assert 0 < float(row["yplus-max"]) < 1.0 and float(row["yplus-avg"]) < float(row["yplus-max"])
    div = next(r for r in summary if r["case"] == "naca6415_Re200000_a16")
    assert div["conv_state"] == "diverged" and div["conv_reason"] == "coefficient out of range"

    # Transition SST로 다시 푼 결과 (Re 2e5의 naca6412만): CL은 받음각이 클수록 차이가 커짐, CD는 20% 작음
    (tmp_path / "runs_tsst").mkdir()
    with (tmp_path / "runs_tsst" / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "state", "cl", "cd", "conv_state"])
        for r in summary:
            if r["case"].startswith("naca6412_Re200000_"):
                w.writerow([r["case"], "done", float(r["cl"]) + 0.02 * (float(r["aoa"]) - 9),
                            0.8 * float(r["cd"]), r["conv_state"]])

    capsys.readouterr()
    assert main(["-c", str(cfg_path), "dataset"]) == 0
    text = capsys.readouterr().out
    assert "SST vs Transition SST: 같은 케이스 12개" in text and "y+ (yplus_max" in text
    out = tmp_path / "runs" / "dataset"
    split = json.loads((out / "split.json").read_text(encoding="utf-8"))["assignments"]
    assert split["naca6412"] == split["naca6415"] == "test_ood"
    rows = list(csv.DictReader((out / "dataset.csv").open(encoding="utf-8-sig")))
    assert len(rows) == len(AIRFOILS) * 12 * 2
    assert {r["split"] for r in rows if r["shape"] == "naca2412"} == {split["naca2412"]}
    assert sum(k.startswith("cst16_") and k[6] in "ul" for k in rows[0]) == 16
    assert sum(k.startswith("cst8_") and k[5] in "ul" for k in rows[0]) == 8
    r = next(r for r in rows if r["case"] == "naca2412_Re200000_a4")
    assert float(r["log10_cd"]) == pytest.approx(math.log10(float(r["cd"])))
    assert {r["near_stall"] for r in rows} == {"0", "1"}
    stall_rows = [r for r in rows if r["shape"] == "naca0012" and r["reynolds"] == "200000.0"]
    clmax_aoa = float(stall_rows[0]["aoa_clmax"])
    assert all((r["near_stall"] == "1") == (abs(float(r["aoa"]) - clmax_aoa) <= 2) for r in stall_rows)
    assert (out / "cst_fits.png").is_file()
    r16 = next(r for r in rows if r["case"] == "naca6412_Re200000_a16")
    assert float(r16["d_cl_tsst"]) == pytest.approx(0.02 * 7)
    assert float(r16["d_log10_cd_tsst"]) == pytest.approx(math.log10(0.8))
    assert float(r16["yplus_max"]) > 0 and r16["avg_iters"] in ("200", "500")
    assert next(r for r in rows if r["case"] == "naca6412_Re500000_a16")["d_cl_tsst"] == ""

    # 다시 만들어도 분할은 그대로
    assert main(["-c", str(cfg_path), "dataset"]) == 0
    assert json.loads((out / "split.json").read_text(encoding="utf-8"))["assignments"] == split

    capsys.readouterr()
    assert main(["-c", str(cfg_path), "train", "--seeds", "2", "--epochs", "300", "--name", "t16"]) == 0
    text = capsys.readouterr().out
    assert "near_stall" in text and "test_ood" in text
    metrics = list(csv.DictReader((tmp_path / "runs" / "ml" / "t16" / "metrics.csv").open(encoding="utf-8-sig")))
    by = {(m["model"], m["split"], m["condition"]): m for m in metrics}
    for model in ("baseline", "lgbm", "mlp"):
        assert ("test_ood" in {m["split"] for m in metrics if m["model"] == model})
    # 학습 모델은 이론식보다 (같은 가짜 물리에서) CD를 훨씬 잘 맞춘다
    assert float(by[("lgbm", "test", "all")]["mae_log10_cd"]) < float(by[("baseline", "test", "all")]["mae_log10_cd"])
    # 이론식의 적용 한계: 부착 유동 CL은 잘 맞지만 실속 후에는 크게 틀림 → 학습 모델이 더 정확
    assert float(by[("baseline", "test", "attached")]["mae_cl"]) < 0.05
    assert float(by[("mlp", "test", "post_stall")]["mae_cl"]) < float(by[("baseline", "test", "post_stall")]["mae_cl"])
    assert "cover90_cl" in by[("mlp", "test", "all")] and "width90_cl" in by[("lgbm", "test", "all")]
    assert any(m["condition"] == "state=oscillating" for m in metrics)
    assert ("mlp", "test_ood", "post_stall") in by   # 깊은 실속(CL 최대 + 창 초과)은 attached와 따로
    # 형상 단위 부트스트랩 95% 신뢰구간
    for key in ("mae_cl", "mae_log10_cd", "cover90_cl"):
        m = by[("mlp", "test", "all")]
        assert float(m[f"{key}_lo95"]) <= float(m[key]) <= float(m[f"{key}_hi95"])
    # 발산은 학습·평가에서 빼고 건수만
    div = by[("lgbm", "test_ood", "state=diverged")]
    assert div["n"] == "1" and div["mae_cl"] == "" and "건수만 보고: val 0, test 0, test_ood 1" in text
    preds = list(csv.DictReader((tmp_path / "runs" / "ml" / "t16" / "predictions.csv").open(encoding="utf-8-sig")))
    assert {"baseline_cl", "lgbm_cl_lo", "mlp_log10_cd_hi", "yplus_max", "d_cl_tsst"} <= set(preds[0])
    assert "naca6415_Re200000_a16" not in {p["case"] for p in preds}

    # 실험 1 (형상 계수 8개), 개선 실험 (처음 보는 형상 일부를 학습에 추가)
    assert main(["-c", str(cfg_path), "train", "--model", "lgbm", "--features", "cst8", "--name", "t8"]) == 0
    run = json.loads((tmp_path / "runs" / "ml" / "t8" / "run.json").read_text(encoding="utf-8"))
    assert len(run["features"]) == 8 + 2
    assert main(["-c", str(cfg_path), "train", "--model", "baseline", "--ood-train-fraction", "0.5",
                 "--name", "ood"]) == 0
    run = json.loads((tmp_path / "runs" / "ml" / "ood" / "run.json").read_text(encoding="utf-8"))
    assert len(run["ood_shapes_moved_to_train"]) == 1

    # 비교: 한 실행 안의 모델끼리 (기준 = 첫 모델)
    assert main(["-c", str(cfg_path), "compare", "t16"]) == 0
    cmp = list(csv.DictReader((tmp_path / "runs" / "ml" / "_compare" / "t16.csv").open(encoding="utf-8-sig")))
    c = {(r["split"], r["condition"], r["model"]): r for r in cmp}
    assert c[("test", "all", "baseline")]["d_mae_cl"] == "" and c[("test", "all", "lgbm")]["reference"] == "t16/baseline"
    assert float(c[("test", "all", "lgbm")]["d_mae_log10_cd_hi95"]) < 0   # LightGBM이 CD를 확실히 더 잘 맞춤
    for key in ("mae_cl", "mae_cl_lo95", "cover90_cl_hi95"):  # 같은 케이스·같은 재표집 → metrics.csv와 같은 값
        assert float(c[("test", "all", "mlp")][key]) == pytest.approx(float(by[("mlp", "test", "all")][key]))
    # 개선 실험과 비교: 학습에 넣은 처음 보는 형상은 양쪽 모두에서 빠짐
    capsys.readouterr()
    assert main(["-c", str(cfg_path), "compare", "t16", "ood", "--model", "baseline", "--name", "imp"]) == 0
    assert "t16: 다른 실행에 없는 케이스 23개 제외" in capsys.readouterr().out
    cmp = list(csv.DictReader((tmp_path / "runs" / "ml" / "_compare" / "imp.csv").open(encoding="utf-8-sig")))
    ood = next(r for r in cmp if (r["run"], r["split"], r["condition"]) == ("ood", "test_ood", "all"))
    assert ood["n_shapes"] == "1" and ood["reference"] == "t16/baseline"
    assert float(ood["d_mae_cl"]) == 0   # 이론식 CL은 학습 데이터와 무관

    # 경계 사례 10개
    capsys.readouterr()
    assert main(["-c", str(cfg_path), "edge-cases", "--run", "t16"]) == 0
    assert "경계 사례 10개" in capsys.readouterr().out
    edge_dir = tmp_path / "runs" / "ml" / "t16" / "edge_cases_mlp"
    edges = list(csv.DictReader((edge_dir / "edge_cases.csv").open(encoding="utf-8-sig")))
    assert len(edges) == 10 and len({e["case"] for e in edges}) == 10
    assert [e["category"] for e in edges] == ["stall_pre"] * 2 + ["stall_post"] * 2 + ["turbulence"] * 2 + [
        "attached_test"] * 2 + ["attached_ood"] * 2
    assert all(e["eval_split"] in ("test", "test_ood") for e in edges if e["category"] != "turbulence")
    assert all(e["eval_split"] == "test_ood" for e in edges if e["category"] == "attached_ood")
    post = next(e for e in edges if e["category"] == "stall_post")
    assert float(post["aoa"]) > float(post["aoa_clmax"])
    assert all(float(e["aoa"]) < float(e["aoa_clmax"]) for e in edges if e["category"].startswith("attached"))
    turb = [e for e in edges if e["category"] == "turbulence"]
    assert all(e["case"].startswith("naca6412_Re200000_") for e in turb)
    assert float(turb[0]["tsst_diff_score"]) >= float(turb[1]["tsst_diff_score"]) > 0
    pair = {e["pick"]: float(e["error_score"]) for e in edges if e["category"] == "attached_test"}
    assert pair["worst"] >= pair["typical"]   # 범주 안에서 최악 사례의 오차 점수가 대표 사례 이상
    assert all((edge_dir / e["figure"]).is_file() for e in edges)


def test_shape_bootstrap_ci_and_pairing():
    from airfoil_surrogate.learn import ShapeBootstrap

    shapes = ["a"] * 5 + ["b"] * 5 + ["c"] * 5
    errs = np.r_[np.full(5, 0.1), np.full(5, 0.2), np.full(5, 0.6)]
    boot = ShapeBootstrap(shapes, "test|all", seed=1)
    ci = boot.mean_ci("mae", errs)
    assert ci["mae"] == pytest.approx(0.3)
    assert 0.1 <= ci["mae_lo95"] < 0.3 < ci["mae_hi95"] <= 0.6   # 형상별 평균(0.1, 0.2, 0.6) 범위 안
    assert ShapeBootstrap(shapes, "test|all", seed=1).mean_ci("mae", errs) == ci   # 같은 시드·키 → 같은 결과
    assert boot.mean_ci("d", errs - errs) == {"d": 0.0, "d_lo95": 0.0, "d_hi95": 0.0}   # 짝지은 재표집
    single = ShapeBootstrap(["a"] * 4, "x", seed=1).mean_ci("m", np.arange(4.0))
    assert single["m_lo95"] == pytest.approx(1.5) and single["m_hi95"] == pytest.approx(1.5)   # 형상 1개면 구간 폭 0
    assert single["m_lo95"] <= single["m"] <= single["m_hi95"]


# ---------------------------------------------------------------- 메쉬 견고성

AIRFOIL_PARAMS = {"main_airfoil": "2412", "chord": 0.3, "reynolds": 5e5}


def test_mesh_fingerprint_unchanged_by_new_settings(tmp_path):
    from airfoil_surrogate.meshing import make_job

    # 지문 값은 AERO_AUTOMATION에서 분리되기 전 코드로 계산한 값과 같다 (분리 전에 만든 메쉬를 다시 만들지 않음).
    # 새 항목(farfield_split, farfield_shape, box_*, timeout)도 기본값이면 지문에 영향 없음
    fp = "69ac648a66c0"
    assert make_job(AIRFOIL_PARAMS, {}, tmp_path).fingerprint() == fp
    assert make_job(AIRFOIL_PARAMS, {"timeout": 10}, tmp_path).fingerprint() == fp
    assert make_job(AIRFOIL_PARAMS, {"farfield_shape": "circle", "box_height": 20}, tmp_path).fingerprint() == fp
    assert make_job(AIRFOIL_PARAMS, {"farfield_split": True}, tmp_path).fingerprint() == "eea950de090f"
    assert make_job(AIRFOIL_PARAMS, {"farfield_shape": "box"}, tmp_path).fingerprint() == "6e17fa42aeec"


def test_cusp_trailing_edge_is_opened(tmp_path):
    from airfoil_surrogate.geometry import CUSP_WEDGE_DEG, split_te_wedge

    xy, le = normalize(naca4("2412", closed_te=True))
    # 뒷전 바로 앞 점 쌍을 거의 겹치게 만든 칼날형 뒷전
    xy[1] = [0.9968, 0.00037]
    xy[-2] = [0.9967, 0.00036]
    path = tmp_path / "cusp.dat"
    path.write_text("cusp\n" + "\n".join(f"{x:.6f} {y:.6f}" for x, y in xy), encoding="utf-8")
    el = load_section("cusp.dat", tmp_path, "main")
    assert len(el.xy) == len(xy) - 2
    assert split_te_wedge(el.xy) >= CUSP_WEDGE_DEG
    assert len(load_section("2412", tmp_path, "main").xy) == len(normalize(naca4("2412"))[0])  # NACA는 그대로


def test_mesh_timeout_fails_only_that_geometry(tmp_path):
    from airfoil_surrogate.config import ConfigError
    from airfoil_surrogate.meshing import ERROR_FILE, generate_mesh, make_job

    job = make_job({"main_airfoil": "0012"}, {"timeout": 0.01}, tmp_path)
    with pytest.raises(ConfigError, match="시간 초과"):
        generate_mesh(job, tmp_path, plots=False)
    assert "시간 초과" in (job.dir / ERROR_FILE).read_text(encoding="utf-8")
    assert not job.mesh_file.exists()


def test_tiny_trailing_edge_gap_is_closed(tmp_path):
    xy, _ = normalize(naca4("2412", closed_te=True))
    xy[-1] = xy[0] + [0.0, 1e-5]  # 반올림으로 생긴 0.00001c 간격 (S1221 좌표 파일과 같은 상황)
    path = tmp_path / "gap.dat"
    path.write_text("gap\n" + "\n".join(f"{x:.7f} {y:.7f}" for x, y in xy), encoding="utf-8")
    el = load_section("gap.dat", tmp_path, "main")
    assert el.sharp_te
    assert not load_section("2412", tmp_path, "main").sharp_te  # 표준 NACA(뒷전 두께 0.0025c)는 그대로
