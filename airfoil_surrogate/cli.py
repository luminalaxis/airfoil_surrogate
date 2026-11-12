"""명령줄 인터페이스 (airfoil-surrogate, 줄여서 afs).

    afs make-cases      # 라틴 하이퍼큐브 (m, p, t, α) → cases.csv, 검증용 cases_validation.csv
    afs fetch-airfoils  # UIUC 좌표 파일 내려받기
    afs plan            # 케이스 목록 확인 + 폴더/저널 생성
    afs mesh            # 형상별 메쉬 생성 + 형상/메쉬 이미지
    afs check           # Fluent 실행 전 점검 (베이스 케이스, 메쉬, 저널, 경로, 디스크·시간)
    afs run             # 배치 실행 (완료된 케이스는 건너뜀)
    afs status          # 케이스별 진행 상태
    afs collect         # 결과 요약 CSV(수렴 상태 포함), 그래프 생성
    afs gci             # 메쉬 독립성 확인 (mesh_level 3단계 → Richardson 외삽, GCI)
    afs dataset         # CST 계수 + 형상 단위 분할 → runs/dataset/
    afs train           # 기준 방법 / LightGBM / MLP 학습·조건별 평가 → runs/ml/
    afs compare A B     # 학습 결과끼리 같은 케이스로 비교 (실험 1·2, 개선 실험)
    afs edge-cases      # 경계 사례 10개 선정 + 사례별 그래프
    afs merge-results   # 팀원별 results.csv 합치기 (공통 스키마 확인)
    afs validate ...    # airfoil2vec (Re 4.66e6) / xfoil (Re 1e5, SST vs SST-LM) / mesh (light vs fine)

모드 전환 (설정 파일의 [flow] 값을 덮어씀, 결과는 runs/<re_mode>_<난류 모델>_<메쉬 모드>_<케이스 목록>/):
    afs --re-mode validation --mesh-mode fine run
    afs --turb kOmegaSSTLM --case-list validation run --assignee member1
"""

from __future__ import annotations

import sys
from collections import Counter

from .banner import print_banner
from .console import ColorArgumentParser, clear_screen
from .cases import build_cases, filter_assignee, filter_cases, partition_cases
from .config import ConfigError, load_config
from .expr import ExpressionError
from .results import collect
from .runner import build_command, ensure_meshes, prepare_case, read_status, run_all


def _cfg(args):
    cfg = load_config(args.config, {"re_mode": args.re_mode, "mesh_mode": args.mesh_mode,
                                    "turbulence_model": args.turb, "case_list": args.case_list})
    if cfg.flow is not None and not getattr(args, "_flow_shown", False):
        args._flow_shown = True
        print(f"[flow] {cfg.flow.describe()}")
        for w in cfg.flow.warnings():
            print(f"[경고] {w}")
    return cfg


def _load(args):
    cfg = _cfg(args)
    cases = filter_assignee(filter_cases(build_cases(cfg), args.only), getattr(args, "assignee", None))
    cases = partition_cases(cases, getattr(args, "part", None))
    if not cases:
        raise ConfigError("조건에 맞는 케이스가 없습니다.")
    return cfg, cases


def _print_table(cases, extra=None) -> None:
    keys = []
    for c in cases:
        keys += [k for k in c.params if k not in keys]
    header = ["case", "group", *keys] + ([extra[0]] if extra else [])
    rows = [
        [c.name, c.group, *(_fmt(c.params.get(k, "")) for k in keys)] + ([extra[1](c)] if extra else [])
        for c in cases
    ]
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    for line in [header, ["-" * w for w in widths], *rows]:
        print("  ".join(str(x).ljust(w) for x, w in zip(line, widths)))


def _fmt(value) -> str:
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def cmd_plan(args) -> int:
    cfg, cases = _load(args)
    _print_table(cases)
    print(f"\n총 {len(cases)}개 케이스 → {cfg.output_dir}")
    meshes = {c.mesh.name for c in cases if c.mesh is not None}
    if meshes:
        print(f"형상(메쉬) {len(meshes)}개 → {cfg.meshes_dir}")
    if args.dry_run:
        return 0
    template_text = cfg.template.read_text(encoding="utf-8")
    for case in cases:
        prepare_case(cfg, case, template_text)
    print("케이스 폴더와 저널 파일(run.jou)을 생성했습니다.")
    try:
        print("실행 명령 예시:", " ".join(build_command(cfg, cases[0])))
    except ConfigError as exc:
        print(f"[경고] {exc}")
    return 0


def _limit_cases(cases, limit: int | None):
    """run --limit N: 남은 케이스 중 앞의 N개만."""
    if limit:
        todo = [c for c in cases if read_status(c).get("state") != "done"]
        cases = todo[:limit]
        print(f"--limit {limit}: 남은 케이스 {len(todo)}개 중 {len(cases)}개만 실행")
    return cases


def cmd_run(args) -> int:
    cfg, cases = _load(args)
    cases = _limit_cases(cases, args.limit)
    print(f"총 {len(cases)}개 케이스 실행 (동시 실행 {args.jobs or cfg.max_parallel}개)")
    results = run_all(cfg, cases, force=args.force, max_parallel=args.jobs)
    counts = Counter(r.state for r in results)
    print("\n결과: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    _print_collect(collect(cfg, build_cases(cfg)), cfg)  # 실행한 케이스 전부 (대기 중은 results.csv에서 빠짐)
    return 1 if counts.get("failed") else 0


def cmd_mesh(args) -> int:
    cfg, cases = _load(args)
    if cfg.mesh is None:
        raise ConfigError("project.toml에 [mesh] 섹션이 없습니다.")
    errors = ensure_meshes(cfg, cases, force=args.force, workers=args.jobs)
    _print_mesh_flags(cases)
    print(f"\n메쉬 폴더: {cfg.meshes_dir}  (형상별 mesh.msh, mesh_info.json, geometry.png, mesh*.png)")
    return 1 if errors else 0


def _print_mesh_flags(cases) -> None:
    import json

    from .meshing import INFO_FILE

    jobs = {c.mesh.name: c.mesh for c in cases if c.mesh is not None}
    bad = []
    for job in jobs.values():
        path = job.dir / INFO_FILE
        if path.is_file():
            info = json.loads(path.read_text(encoding="utf-8"))
            if info.get("quality_flag") == "bad":
                bad.append((job.name, info.get("quality_reasons", [])))
    if bad:
        print(f"\n[경고] 불량 판정 메쉬 {len(bad)}개 (메쉬는 남기고 표시만, results의 mesh_flag 열):")
        for name, reasons in bad[:10]:
            print(f"  {name}: {'; '.join(reasons)}")


def _print_collect(result: dict, cfg) -> None:
    print(f"요약: {result['summary']}")
    if result.get("results"):
        print(f"공통 스키마 결과: {result['results']}  ({result['n_results']}개 케이스, converged {result['n_converged']}개)")
    if result["ranking"]:
        key = "ld_max" if cfg.results.objective == "max_ld" else "cl_max"
        print(f"형상 순위 ({key} 기준): {result['ranking']}")
        for e in result["top"]:
            desc = ", ".join(f"{k}={v}" for k, v in e.items()
                             if k not in ("rank", "n_aoa", "mesh") and "@" not in k and not k.endswith("_max"))
            extra = f"L/D max {e['ld_max']:.2f}" if "ld_max" in e else ""
            print(f"  {e['rank']}. {desc}  Cl max {e['cl_max']:.3f}  {extra}")
    if result["plots"]:
        print(f"그래프: {result['plots'][0].parent}")
    for w in result.get("warnings", []):
        print(f"[경고] {w}")


def cmd_check(args) -> int:
    from .preflight import ERROR, TODO, format_items, run_checks

    cfg, cases = _load(args)
    items = run_checks(cfg, cases, parallel=args.jobs)
    print(format_items(items))
    return 1 if any(it.level in (ERROR, TODO) for it in items) else 0


def cmd_status(args) -> int:
    cfg, cases = _load(args)
    statuses = {c.name: read_status(c) for c in cases}

    def state(c):
        s = statuses[c.name]
        return s["state"] + (f" ({s['message']})" if s.get("message") else "")

    _print_table(cases, ("state", state))
    counts = Counter(s["state"] for s in statuses.values())
    print("\n" + ", ".join(f"{k} {v}" for k, v in counts.items()))
    eta = _eta(list(statuses.values()), args.jobs or cfg.max_parallel)
    if eta:
        print(eta)
    return 0


def _eta(statuses: list[dict], parallel: int) -> str | None:
    """끝난 케이스의 실행 시간(중앙값)으로 남은 케이스(대기·실행 중·실패)의 예상 소요 시간."""
    import statistics

    times = [s["elapsed_sec"] for s in statuses if s.get("state") == "done" and s.get("elapsed_sec")]
    left = sum(1 for s in statuses if s.get("state") in ("pending", "running", "failed", "interrupted", "unknown"))
    if not times or not left:
        return None
    per_case = statistics.median(times)
    hours = per_case * left / max(1, parallel) / 3600
    return (f"남은 케이스 {left}개 × 케이스당 {per_case / 60:.1f}분(완료 {len(times)}개 중앙값) "
            f"÷ 동시 {max(1, parallel)}개 ≈ {hours:.1f}시간 ({hours / 24:.1f}일)")


def cmd_collect(args) -> int:
    cfg, cases = _load(args)
    _print_collect(collect(cfg, cases, plots=not args.no_plots), cfg)
    return 0


def cmd_gci(args) -> int:
    import math

    from .mesh_study import mesh_study

    cfg, cases = _load(args)
    r = mesh_study(cfg, cases, quantities=args.quantity, plots=not args.no_plots)
    print("메쉬 독립성 (Celik et al. 2008 GCI): 1 = 조밀, 2 = 보통, 3 = 성김 (셀 수 기준)")
    print("GCI = 그 메쉬 결과의 이산화 불확실성 (95% 신뢰 수준 추정), 외삽 = 셀 크기 0으로 외삽한 값\n")
    print(f"{'group':<58}{'q':<9}{'조밀':>10}{'보통':>10}{'성김':>10}{'p':>6}{'외삽':>10}"
          f"{'GCI 조밀':>9}{'GCI 보통':>9}  판정")
    last = None
    for e in r["rows"]:
        group = e["group"] if e["group"] != last else ""
        last = e["group"]

        def pct(v):
            return f"{v * 100:.2f}%" if isinstance(v, float) and math.isfinite(v) else "-"

        def num(v, fmt=".4f"):
            return format(v, fmt) if isinstance(v, float) and math.isfinite(v) else "-"

        print(f"{group[:57]:<58}{e['quantity']:<9}{e['phi_fine']:>10.4f}{e['phi_medium']:>10.4f}"
              f"{e['phi_coarse']:>10.4f}{num(e.get('p'), '.2f'):>6}{num(e.get('phi_ext')):>10}"
              f"{pct(e.get('gci_fine')):>9}{pct(e.get('gci_medium')):>9}  {e['convergence']}")
    for g in r["incomplete"]:
        print(f"[참고] {g['group']}: 끝난 메쉬 단계 {g['levels']}개 (3단계가 모두 끝나야 계산)")
    bad = [e for e in r["rows"] if e["quantity"] in ("cl", "cd") and not (
        isinstance(e.get("gci_medium"), float) and math.isfinite(e["gci_medium"])
        and e["gci_medium"] <= (0.02 if e["quantity"] == "cl" else 0.05))]
    if r["rows"]:
        if bad:
            print(f"\n[판단] 보통(본 배치) 메쉬의 GCI가 기준(Cl 2%, Cd 5%)을 넘거나 수렴이 단조롭지 않은 항목 {len(bad)}개."
                  " 메쉬를 조밀하게(project.toml [mesh] refine 1.3~1.5 또는 surface_cells·near_size) 하거나,"
                  " 반복 수렴(conv_state)부터 확인하세요.")
        else:
            print("\n[판단] 보통(본 배치) 메쉬의 GCI가 Cl 2%, Cd 5% 이내입니다. 현재 [mesh] 설정으로 본 배치를 진행해도 됩니다.")
    print(f"\n결과: {r['csv']}" + (f"  (그래프 {len(r['plots'])}개)" if r["plots"] else ""))
    return 0


def cmd_make_cases(args) -> int:
    from .sampling import make_cases

    cfg = _cfg(args)
    r = make_cases(cfg, force=args.force)
    for key, v in r.items():
        rows = v["rows"]
        label = "본 데이터" if key == "csv" else "검증용"
        print(f"{label}: {v['path']}  케이스 {len(rows)}개, 형상 {len({x['shape_id'] for x in rows})}개")
        for who, (n_shapes, n_cases) in sorted(v["by_assignee"].items()):
            print(f"  {who:<12} 형상 {n_shapes:>4}개, 케이스 {n_cases:>5}개")
    print("같은 설정(seed)이면 누가 실행해도 같은 파일이 나옵니다. 팀원은 각자 afs run --assignee <이름>.")
    return 0


def cmd_merge_results(args) -> int:
    from pathlib import Path

    from .results import merge_results

    r = merge_results([Path(p) for p in args.inputs], Path(args.output))
    print(f"합침: 파일 {r['files']}개 → {r['path']}  ({r['rows']}개 케이스, converged {r['converged']}개)")
    return 0


def _pct(v) -> str:
    import math

    return f"{v * 100:.1f}%" if isinstance(v, float) and math.isfinite(v) else "-"


def cmd_validate(args) -> int:
    import math
    from pathlib import Path

    from . import validate as v

    cfg = _cfg(args)
    if args.stage == "airfoil2vec":
        r = v.validate_airfoil2vec(cfg, [Path(p) for p in args.results] if args.results else None, args.metadata)
        s = r["summary"]
        print(f"Airfoil2Vec metadata: {s['metadata']}  (형상 {r['reference_shapes']}개)")
        print(f"판정 기준: CL {s['criteria']['CL']}, CD {s['criteria']['CD']}  "
              "(우리 +α ↔ 그쪽 inverted −α, CL·CM 부호 반전)")
        ok_all = True
        for mode, m in s["by_mesh_mode"].items():
            ok = m["judged"] > 0 and m["pass"] == m["judged"]
            ok_all &= ok
            print(f"  [{mode}] 비교 {m['judged']}/{m['cases']}개, 통과 {m['pass']} (CL {m['CL_pass']}, CD {m['CD_pass']}), "
                  f"|ΔCL| 평균 {m['mean_abs_dCL']:.4f}, |ΔCD|/CD 평균 {_pct(m['mean_abs_dCD_rel'])} "
                  f"최대 {_pct(m['max_abs_dCD_rel'])}" + (f", 미수렴 {m['not_converged']}" if m["not_converged"] else "")
                  + ("  → 통과" if ok else "  → 미통과"))
        d = s.get("light_vs_fine")
        if d:
            print(f"  경량 메쉬 오차 (light − fine, {d['cases']}개): |ΔCL| 평균 {d['CL']['mean_abs']:.4f} 최대 "
                  f"{d['CL']['max_abs']:.4f}, |ΔCD|/CD 평균 {_pct(d['CD']['mean_abs_rel'])} 최대 {_pct(d['CD']['max_abs_rel'])}")
        unmatched = [x for x in r["rows"] if not x.get("matched")]
        if unmatched:
            print(f"[경고] 기준 데이터에서 짝을 못 찾은 케이스 {len(unmatched)}개 (예: {unmatched[0]['reference']} "
                  f"α {-unmatched[0]['alpha']:g}). name_format·열 이름을 확인하세요.")
        print(f"결과: {r['dir']}  (airfoil2vec_report.csv, airfoil2vec_summary.json, mesh_light_vs_fine.csv)")
        return 0 if ok_all else 1
    if args.stage == "xfoil":
        r = v.validate_xfoil(cfg)
        s = r["summary"]
        print(f"XFOIL: Re {s['Re']:,.0f}, Ncrit {s['ncrit']:g}, 형상 {s['shapes']}개, 수렴한 점 {s['xfoil_points']}개")
        for model, m in s["by_model"].items():
            if "error" in m:
                print(f"  {model:<12} {m['error']}")
                continue
            dcm = m["mean_abs_dCM"]
            dcm_txt = f"{dcm:.4f}" if isinstance(dcm, float) and math.isfinite(dcm) else "-"
            print(f"  {model:<12} 비교 {m['compared']}개: |ΔCL| 평균 {m['mean_abs_dCL']:.4f}, "
                  f"|ΔCD|/CD 평균 {_pct(m['mean_abs_dCD_rel'])}, |ΔCM| 평균 {dcm_txt}, 참고 판정 통과 {m['pass']}")
        if s["recommended"]:
            print(f"XFOIL에 더 가까운 난류 모델: {s['recommended']}  (점수 = |ΔCL|/0.1 + |ΔCD/CD|)")
        print(f"결과: {r['dir']}  (xfoil_polars.csv, xfoil_compare.csv, xfoil_summary.json)")
        return 0
    r = v.validate_mesh_modes(cfg)
    d = r["summary"]
    print(f"light − fine ({d['cases']}개): |ΔCL| 평균 {d['CL']['mean_abs']:.4f} 최대 {d['CL']['max_abs']:.4f}, "
          f"|ΔCD|/CD 평균 {_pct(d['CD']['mean_abs_rel'])} 최대 {_pct(d['CD']['max_abs_rel'])}")
    print(f"결과: {r['path']}")
    return 0


def cmd_fetch_airfoils(args) -> int:
    from pathlib import Path

    from .airfoil_db import UIUC_URL, UIUC_ZIP_URL, fetch_airfoils

    cfg, cases = _load(args)
    r = fetch_airfoils(cfg, cases, url=args.url or UIUC_URL,
                       zip_path=Path(args.zip).resolve() if args.zip else None, force=args.force)
    print(f"받음 {len(r['ok'])}개, 이미 있음 {len(r['skipped'])}개, 실패 {len(r['failed'])}개")
    for spec, reason in r["failed"].items():
        print(f"  [실패] {spec}: {reason}")
    if r["failed"]:
        print(f"사이트 접속이 안 되면 {UIUC_ZIP_URL} 을 받아 --zip 으로 지정하세요.")
    return 1 if r["failed"] else 0


def cmd_dataset(args) -> int:
    from .dataset import build_dataset

    cfg, cases = _load(args)
    r = build_dataset(cfg, cases, resplit=args.resplit)
    print(f"데이터셋: {r['dir']}  (dataset.csv, airfoils.csv, split.json, cst_fits.png)")
    print("분할       형상   케이스")
    for split, c in r["counts"].items():
        print(f"  {split:<9} {c['shapes']:>4}  {c['cases']:>6}")
    print("수렴 상태: " + ", ".join(f"{k} {v}" for k, v in sorted(r["states"].items()))
          + "  (diverged는 학습·평가에서 빼고 건수만 보고)")
    print(f"실속 부근(CL 최대 받음각 ± 창) 케이스: {r['near_stall']}")
    if r["yplus"]:
        y = r["yplus"]
        print(f"y+ (yplus_max, 학습에 쓰는 케이스 {y['n']}개): 중앙값 {y['median']:.2f}, 최대 {y['max']:.2f}")
    if r["tsst"] is not None:
        t = r["tsst"]
        print(f"SST vs Transition SST: 같은 케이스 {t['n']}개 비교 (차이 = T-SST − SST)")
        for re, st in t["by_re"].items():
            print(f"  Re {re:,.0f}: {st['n']}개, |ΔCL| 중앙값 {st['median_abs_d_cl']:.3f} · 최대 {st['max_abs_d_cl']:.3f}"
                  f" ({st['max_d_cl_case']}), |Δlog10 CD| 중앙값 {st['median_abs_d_log10_cd']:.3f}"
                  f" · 최대 {st['max_abs_d_log10_cd']:.3f} ({st['max_d_log10_cd_case']})")
        if not t["n"]:
            print("  [경고] 이름이 같은 케이스가 없습니다. Transition SST 프로젝트도 같은 케이스 이름 패턴을 쓰세요.")
    for w in r["warnings"]:
        print(f"[경고] {w}")
    return 0


def _ci(m: dict, key: str, fmt: str = ".4f") -> str:
    """값 [95% 신뢰구간]."""
    if m.get(key) is None:
        return "-"
    if m.get(f"{key}_lo95") is None:
        return format(m[key], fmt)
    return f"{m[key]:{fmt}} [{m[f'{key}_lo95']:{fmt}}, {m[f'{key}_hi95']:{fmt}}]"


def cmd_train(args) -> int:
    from .learn import PI_TAG, TrainOptions, train

    cfg = _cfg(args)
    opts = TrainOptions(models=args.model, features=args.features,
                        hidden=[int(h) for h in args.hidden.split(",") if h],
                        n_seeds=args.seeds, epochs=args.epochs, stall_weight=args.stall_weight,
                        ood_train_fraction=args.ood_train_fraction, name=args.name)
    r = train(cfg, opts)
    print(f"학습 케이스 {r['n_train']}개, 검증 {r['n_val']}개, 입력 {len(r['features'])}개 ({opts.features} + aoa + log10_re)")
    if r["moved"]:
        print(f"test_ood 형상 {len(r['moved'])}개를 학습에 추가: {', '.join(r['moved'])}")
    print(f"[ ] = 형상 단위 부트스트랩 95% 신뢰구간, PI{PI_TAG} = {PI_TAG}% 예측 구간 (CL)")
    print(f"\n{'model':<9}{'split':<10}{'condition':<12}{'n':>6}  {'MAE CL [95% CI]':<27}{'MAE log10CD [95% CI]':<27}"
          f"{'PI' + PI_TAG + ' width':>11}  {'PI' + PI_TAG + ' cover [95% CI]':<20}")
    for m in r["metrics"]:
        if m["condition"].startswith("state="):
            continue
        width = f"{m[f'width{PI_TAG}_cl']:.3f}" if f"width{PI_TAG}_cl" in m else "-"
        print(f"{m['model']:<9}{m['split']:<10}{m['condition']:<12}{m['n']:>6}  {_ci(m, 'mae_cl', '.3f'):<27}"
              f"{_ci(m, 'mae_log10_cd', '.3f'):<27}{width:>11}  {_ci(m, f'cover{PI_TAG}_cl', '.2f'):<20}")
    if any(r["diverged"].values()):
        print("\n발산(diverged) 케이스는 학습·평가에서 빼고 건수만 보고: "
              + ", ".join(f"{s} {n}" for s, n in r["diverged"].items()))
    print(f"\n결과: {r['dir']}  (metrics.csv: 수렴 상태별 포함, predictions.csv, run.json)")
    return 0


def _delta(e: dict, key: str) -> str:
    """기준 대비 차이 [95% 신뢰구간], 구간이 0을 포함하지 않으면 *."""
    if e.get(key) is None:
        return "(기준)"
    lo, hi = e[f"{key}_lo95"], e[f"{key}_hi95"]
    return f"{e[key]:+.4f} [{lo:+.4f}, {hi:+.4f}]" + ("*" if lo > 0 or hi < 0 else "")


def cmd_compare(args) -> int:
    from .analysis import compare
    from .learn import PI_TAG

    cfg = _cfg(args)
    r = compare(cfg, args.runs, args.model, args.name)
    print(f"모든 실행에 공통인 평가 케이스 {r['n_cases']}개(test, test_ood)로 비교합니다.")
    print("ΔMAE = MAE − 기준 MAE (음수 = 개선), [ ] = 짝지은 형상 단위 부트스트랩 95% 신뢰구간, "
          "* = 구간이 0을 포함하지 않음")
    for run, n in r["not_common"].items():
        if n:
            print(f"  {run}: 다른 실행에 없는 케이스 {n}개 제외 (학습에 넣은 처음 보는 형상 등)")
    if r["mismatch"]:
        print(f"[경고] 실행마다 정답·분할이 다른 케이스 {len(r['mismatch'])}개 제외 "
              f"(사이에 dataset을 다시 만든 것 같습니다): {', '.join(r['mismatch'][:5])}")
    print(f"\n{'split':<10}{'condition':<12}{'n':>6}  {'run/model':<26}{'MAE CL':>8}  {'ΔMAE CL [95% CI]':<30}"
          f"{'MAE logCD':>10}  {'ΔMAE logCD [95% CI]':<30}{'PI' + PI_TAG + ' cover CL':>15}")
    for e in r["rows"]:
        cover = e.get(f"cover{PI_TAG}_cl")
        print(f"{e['split']:<10}{e['condition']:<12}{e['n']:>6}  {e['run'] + '/' + e['model']:<26}"
              f"{e['mae_cl']:>8.4f}  {_delta(e, 'd_mae_cl'):<30}{e['mae_log10_cd']:>10.4f}  "
              f"{_delta(e, 'd_mae_log10_cd'):<30}{'-' if cover is None else f'{cover:.0%}':>15}")
    print(f"\n결과: {r['path']}")
    return 0


def cmd_edge_cases(args) -> int:
    from .analysis import edge_cases
    from .learn import PI_TAG

    cfg = _cfg(args)
    r = edge_cases(cfg, args.run, args.model, plots=not args.no_plots)
    s = r["scale"]
    print(f"경계 사례 {len(r['rows'])}개 ({args.run}, {args.model}). 오차 점수 = |CL 오차| / {s['cl']:.4f} "
          f"+ |log10 CD 오차| / {s['log10_cd']:.4f}  (평가 세트 평균 오차의 몇 배인지, 1 = 평균)")
    print(f"PI{PI_TAG}: CFD 정답이 {PI_TAG}% 예측 구간 안이면 O (CL/log10 CD)\n")
    print(f"{'no':>3}  {'category':<14}{'pick':<13}{'case':<34}{'split':<10}{'CL err':>8}{'logCD err':>10}"
          f"{'score':>7}{'PI' + PI_TAG:>7}")
    for e in r["rows"]:
        if f"in_pi{PI_TAG}_cl" in e:
            pi = "/".join("O" if e.get(f"in_pi{PI_TAG}_{t}") else "X" for t in ("cl", "log10_cd"))
        else:
            pi = "-"
        extra = ""
        if e["category"] == "turbulence":
            extra = f"   T-SST − SST: ΔCL {e['d_cl_tsst']:+.3f}, Δlog10 CD {e['d_log10_cd_tsst']:+.3f}"
        print(f"{e['no']:>3}  {e['category']:<14}{e['pick']:<13}{e['case']:<34}{e['eval_split']:<10}"
              f"{e['err_cl']:>+8.3f}{e['err_log10_cd']:>+10.3f}{e['error_score']:>7.1f}{pi:>7}{extra}")
    if not r["has_tsst"]:
        print("\n[참고] Transition SST 결과가 없어 난류 모델 민감 사례 2개는 고르지 못했습니다. "
              "Transition SST로 일부 케이스를 다시 푼 뒤 [ml] tsst_summary를 지정하고 dataset → train → edge-cases 순서로 다시 실행하세요.")
    other = [c for c in r["missing"] if c != "turbulence"]
    if other:
        print(f"[참고] 해당하는 평가 케이스가 없는 범주: {', '.join(other)}")
    print(f"\n결과: {r['dir']}  (edge_cases.csv, 사례별 CL-α · log10 CD-α 그래프)")
    return 0


def _safe_console() -> None:
    """출력을 파일로 넘길 때(예: afs run > log.txt) 한국어 Windows 기본 인코딩(cp949)에 없는
    문자(−, ≈ 등)가 나와도 멈추지 않도록, 표현할 수 없는 문자는 ?로 바꿔 쓴다."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv=None) -> int:
    parser = ColorArgumentParser(prog="afs", description="단일 익형 Fluent 데이터셋 생성과 CL/CD 회귀 모델")
    parser.add_argument("-c", "--config", default="project.toml", help="프로젝트 설정 파일 (기본: project.toml)")
    parser.add_argument("--no-banner", action="store_true",
                        help="시작 배너 숨기기 (환경 변수 AFS_NO_BANNER=1 과 같음)")
    parser.add_argument("--no-clear", action="store_true",
                        help="시작할 때 터미널 화면을 지우지 않기 (환경 변수 AFS_NO_CLEAR=1 과 같음)")
    parser.add_argument("--re-mode", choices=["production", "validation"],
                        help="[flow] re_mode 덮어쓰기: production (Re 1e5) / validation (Re 4.66e6)")
    parser.add_argument("--mesh-mode", choices=["light", "fine"], help="[flow] mesh_mode 덮어쓰기")
    parser.add_argument("--turb", choices=["kOmegaSST", "kOmegaSSTLM"],
                        help="[flow] turbulence_model 덮어쓰기 (Fluent SST k-ω / Transition SST)")
    parser.add_argument("--case-list", choices=["main", "validation"],
                        help="케이스 목록: main (cases.csv) / validation (cases_validation.csv)")
    sub = parser.add_subparsers(dest="command")

    def add(name, func, help_text):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--only", nargs="+", metavar="PATTERN",
                       help="케이스/그룹 이름 필터 (와일드카드 가능, 예: 'M0.3*')")
        p.add_argument("--part", metavar="K/N",
                       help="형상 단위로 N등분한 K번째 몫만 (팀원 분담용, 예: 1/4)")
        p.add_argument("--assignee", nargs="+", metavar="NAME",
                       help="cases.csv의 assignee(담당자)가 이 이름인 케이스만 (예: --assignee member1)")
        p.set_defaults(func=func)
        return p

    p = add("plan", cmd_plan, "케이스 목록 확인 및 폴더/저널 생성")
    p.add_argument("--dry-run", action="store_true", help="목록만 출력하고 파일은 만들지 않음")
    p = add("mesh", cmd_mesh, "형상별 메쉬 생성 (이미 있으면 건너뜀)")
    p.add_argument("--force", action="store_true", help="기존 메쉬도 다시 생성")
    p.add_argument("-j", "--jobs", type=int, help="동시에 만들 메쉬 수 (기본: CPU 코어 수, 최대 8. 메쉬당 메모리 약 300 MB)")
    p = add("check", cmd_check, "Fluent 실행 전 점검 (베이스 케이스, 메쉬, 저널, 경로, 디스크·시간 추정)")
    p.add_argument("-j", "--jobs", type=int, help="시간 추정에 쓸 동시 실행 수 (기본: [project] max_parallel)")
    p = add("run", cmd_run, "배치 실행 (필요한 메쉬 자동 생성)")
    p.add_argument("--force", action="store_true", help="완료된 케이스도 다시 실행")
    p.add_argument("-j", "--jobs", type=int, help="동시 실행 케이스 수 (라이선스 수 고려)")
    p.add_argument("--limit", type=int, metavar="N", help="남은 케이스 중 앞의 N개만 실행 (나눠서 돌릴 때)")
    p = add("status", cmd_status, "진행 상태 확인 (완료된 케이스 실행 시간으로 남은 시간 추정)")
    p.add_argument("-j", "--jobs", type=int, help="남은 시간 추정에 쓸 동시 실행 수 (기본: [project] max_parallel)")
    p = add("collect", cmd_collect, "결과 요약 CSV, 형상 순위, 그래프 생성")
    p.add_argument("--no-plots", action="store_true", help="그래프 없이 CSV만 (케이스가 수천 개일 때 빠름)")
    p = add("gci", cmd_gci, "메쉬 독립성 확인: mesh_level 3단계 결과로 Richardson 외삽·GCI 계산")
    p.add_argument("--quantity", nargs="+", metavar="Q", help="계산할 값 (기본: cl, cd와 요소별 cl-*, cd-*)")
    p.add_argument("--no-plots", action="store_true", help="그래프 없이 CSV만")
    p = sub.add_parser("make-cases", help="라틴 하이퍼큐브 (m, p, t, α) 케이스 목록 → cases.csv, cases_validation.csv")
    p.add_argument("--force", action="store_true", help="내용이 다른 기존 목록도 덮어씀")
    p.set_defaults(func=cmd_make_cases)
    p = sub.add_parser("merge-results", help="팀원별 results.csv를 하나로 합치기 (공통 스키마·중복 확인)")
    p.add_argument("output", help="합친 파일 경로")
    p.add_argument("inputs", nargs="+", help="results.csv 파일들")
    p.set_defaults(func=cmd_merge_results)
    p = sub.add_parser("validate", help="검증: airfoil2vec (Re 4.66e6) / xfoil (Re 1e5, SST vs SST-LM) / mesh (light vs fine)")
    p.add_argument("stage", choices=["airfoil2vec", "xfoil", "mesh"])
    p.add_argument("--results", nargs="+", help="(airfoil2vec) 비교할 results.csv (기본: validation 모드 light·fine 결과)")
    p.add_argument("--metadata", help="(airfoil2vec) metadata parquet/csv 경로 ([validation.airfoil2vec] metadata 대신)")
    p.set_defaults(func=cmd_validate)
    p = add("fetch-airfoils", cmd_fetch_airfoils, "UIUC 좌표 파일 내려받기 (없는 파일만)")
    p.add_argument("--zip", help="UIUC coord_seligFmt.zip 경로 (사이트 대신 압축 파일에서 꺼냄)")
    p.add_argument("--url", help="내려받을 주소 템플릿 (기본: UIUC, {name} = 파일 이름)")
    p.add_argument("--force", action="store_true", help="이미 있는 파일도 다시 받기")
    p = add("dataset", cmd_dataset, "학습용 데이터셋 (CST 계수, 수렴 상태, 형상 단위 분할)")
    p.add_argument("--resplit", action="store_true", help="split.json을 무시하고 형상 분할을 새로 함")
    p = sub.add_parser("train", help="회귀 모델 학습·평가 (dataset 결과 사용)")
    p.add_argument("--model", nargs="+", default=["baseline", "lgbm", "mlp"],
                   choices=["baseline", "lgbm", "mlp"], help="학습할 모델 (기본: 셋 다)")
    p.add_argument("--features", default="cst16", choices=["cst16", "cst8", "mpt"],
                   help="형상 특성: cst16(CST 면당 8개), cst8(면당 4개), mpt(NACA m, p, t)")
    p.add_argument("--hidden", default="64,64", help="MLP 은닉층 크기 (예: 32,32 / 128,128,128)")
    p.add_argument("--seeds", type=int, default=5, help="MLP 앙상블 시드 수")
    p.add_argument("--epochs", type=int, default=2000, help="MLP 최대 에폭 (검증 손실로 조기 종료)")
    p.add_argument("--stall-weight", type=float, default=1.0, help="실속 부근 학습 케이스 가중치 (개선 실험)")
    p.add_argument("--ood-train-fraction", type=float, default=0.0,
                   help="처음 보는 형상(test_ood) 중 학습에 추가할 비율 (개선 실험)")
    p.add_argument("--name", help="결과 폴더 이름 (기본: 특성_은닉층)")
    p.set_defaults(func=cmd_train)
    p = sub.add_parser("compare", help="학습 결과끼리 같은 케이스·같은 지표로 비교 (실험 1·2, 개선 실험)")
    p.add_argument("runs", nargs="+", metavar="RUN", help="runs/ml/ 아래 결과 폴더 이름. 첫 번째가 기준")
    p.add_argument("--model", nargs="+", choices=["baseline", "lgbm", "mlp"],
                   help="비교할 모델 (기본: 결과에 있는 모델 전부)")
    p.add_argument("--name", help="결과 파일 이름 (기본: 실행 이름을 _vs_로 이음)")
    p.set_defaults(func=cmd_compare)
    p = sub.add_parser("edge-cases", help="경계 사례 10개 선정 + 사례별 그래프 (train 결과 사용)")
    p.add_argument("--run", default="cst16_h64x64", help="runs/ml/ 아래 결과 폴더 이름 (기본: cst16_h64x64)")
    p.add_argument("--model", default="mlp", choices=["baseline", "lgbm", "mlp"], help="분석할 모델 (기본: mlp)")
    p.add_argument("--no-plots", action="store_true", help="그래프 없이 CSV만")
    p.set_defaults(func=cmd_edge_cases)

    _safe_console()
    # 도움말(-h)도 지운 화면에 나오도록 인자 해석 전에 지운다 (터미널일 때만)
    if "--no-clear" not in (sys.argv[1:] if argv is None else argv):
        clear_screen()
    args = parser.parse_args(argv)
    if args.command is None:  # 명령 없이 실행하면 배너 + 도움말
        if not args.no_banner:
            print_banner()
        parser.print_help()
        return 0
    if not args.no_banner:
        print_banner()
    try:
        return args.func(args)
    except (ConfigError, ExpressionError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n중단되었습니다. 다시 run 하면 완료된 케이스는 건너뜁니다.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
