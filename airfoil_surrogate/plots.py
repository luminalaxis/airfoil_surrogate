"""이미지 자동 저장: 형상, 메쉬, 수렴 이력, Cl/Cd 요약 그래프."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

DPI = 150


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_geometry(geom, path: Path, title: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for el in geom.elements:
        ax.fill(el.xy[:, 0], el.xy[:, 1], alpha=0.25)
        ax.plot(el.xy[:, 0], el.xy[:, 1], lw=1, label=el.name)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    return _save(fig, path)


def plot_mesh(nodes: np.ndarray, cells: list, geom, out_dir: Path, title: str = "") -> list[Path]:
    """근거리 뷰 + 앞전 확대 뷰 + 전체 해석 영역."""
    allxy = np.vstack([el.xy for el in geom.elements])
    (x0, y0), (x1, y1) = allxy.min(0), allxy.max(0)
    views = [("mesh.png", (x0 - 0.3, x1 + 0.6), (y0 - 0.35, y1 + 0.35)),
             ("mesh_le.png", (-0.03, 0.05), (-0.035, 0.035))]
    # 전체 해석 영역 (바깥 경계 모양과 크기 확인용)
    (nx0, ny0), (nx1, ny1) = nodes.min(0), nodes.max(0)
    pad = 0.02 * max(nx1 - nx0, ny1 - ny0)
    views.append(("mesh_domain.png", (nx0 - pad, nx1 + pad), (ny0 - pad, ny1 + pad)))

    polys = [nodes[list(c)] for c in cells]
    saved = []
    for fname, xlim, ylim in views:
        xs = [p[:, 0] for p in polys]
        # 보이는 영역의 셀만 그려서 속도 확보
        visible = [p for p, x in zip(polys, xs)
                   if x.max() >= xlim[0] and x.min() <= xlim[1]
                   and p[:, 1].max() >= ylim[0] and p[:, 1].min() <= ylim[1]]
        fig, ax = plt.subplots(figsize=(9, 9 * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])))
        ax.add_collection(PolyCollection(visible, facecolors="none", edgecolors="k", linewidths=0.15))
        for el in geom.elements:
            ax.fill(el.xy[:, 0], el.xy[:, 1], color="0.7")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_title(f"{title}  ({len(cells):,} cells)", fontsize=9)
        saved.append(_save(fig, out_dir / fname))
    return saved


def plot_convergence(report_files: list[Path], path: Path, title: str = "") -> Path | None:
    from .results import parse_report_file

    series = []
    for f in report_files:
        headers, rows = parse_report_file(f)
        if rows:
            arr = np.array(rows)
            series += [(h, arr[:, 0], arr[:, i]) for i, h in enumerate(headers) if i > 0]
    if not series:
        return None
    n = len(series)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.0 * n), sharex=True, squeeze=False)
    for ax, (name, it, val) in zip(axes[:, 0], series):
        ax.plot(it, val, lw=1)
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
        # 마지막 20% 구간 기준으로 y축을 잡아 수렴 상태가 보이게
        tail = val[int(len(val) * 0.8):]
        if len(tail) > 2 and np.ptp(tail) > 0:
            pad = max(np.ptp(tail) * 2, abs(tail.mean()) * 0.02)
            ax.set_ylim(tail.mean() - pad, tail.mean() + pad)
    axes[-1, 0].set_xlabel("iteration")
    axes[0, 0].set_title(title, fontsize=9)
    return _save(fig, path)


def plot_summary(rows: list[dict], out_dir: Path, x_param: str, group_keys: list[str],
                 cl_label: str = "Cl") -> list[Path]:
    """형상별 Cl-α, Cd-α, L/D-α, 극곡선(Cl-Cd)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("state") == "done" and isinstance(r.get("cl"), float) and x_param in r:
            label = ", ".join(f"{k}={r[k]}" for k in group_keys if k in r) or "all"
            groups[label].append(r)
    if not groups:
        return []

    def figure(ylabel, xlabel=x_param):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        return fig, ax

    specs = [
        ("cl_alpha.png", cl_label, lambda r: (r[x_param], r["cl"])),
        ("cd_alpha.png", "Cd", lambda r: (r[x_param], r.get("cd"))),
        ("ld_alpha.png", "L/D", lambda r: (r[x_param], r.get("l/d"))),
        ("polar.png", cl_label, lambda r: (r.get("cd"), r["cl"])),
    ]
    saved = []
    for fname, ylabel, xy in specs:
        fig, ax = figure(ylabel, "Cd" if fname == "polar.png" else x_param)
        for label, rs in sorted(groups.items()):
            pts = sorted((p for p in map(xy, sorted(rs, key=lambda r: r[x_param]))
                          if p[0] is not None and p[1] is not None),
                         key=lambda p: p[0] if fname != "polar.png" else 0)
            if pts:
                ax.plot(*zip(*pts), "o-", ms=3, lw=1, label=label)
        if len(groups) <= 12:
            ax.legend(fontsize=7)
        else:
            ax.set_title(f"{len(groups)} geometries (legend omitted)", fontsize=9)
        saved.append(_save(fig, out_dir / fname))
    return saved


def plot_condition_trend(ranking: list[dict], cond_key: str, geo_keys: list[str], objective: str,
                         path: Path, top: int = 10) -> Path | None:
    """조건(예: 속도)에 따른 형상별 최대 L/D(또는 Cl) 변화. 평균 상위 형상만 표시."""
    metric = "ld_max" if objective == "max_ld" else "cl_max"
    series: dict[str, list[tuple]] = defaultdict(list)
    for e in ranking:
        if isinstance(e.get(cond_key), (int, float)) and e.get(metric) is not None:
            label = ", ".join(f"{k}={e[k]}" for k in geo_keys if k in e) or "all"
            series[label].append((e[cond_key], e[metric]))
    if not series or all(len(v) < 2 for v in series.values()):
        return None
    best = sorted(series, key=lambda s: -np.mean([v for _, v in series[s]]))[:top]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label in best:
        pts = sorted(series[label])
        ax.plot(*zip(*pts), "o-", ms=4, lw=1.2, label=label)
    ax.set_xlabel(cond_key)
    ax.set_ylabel({"ld_max": "max L/D", "cl_max": "max Cl"}[metric])
    ax.set_title(f"top {len(best)} geometries by mean {metric}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    return _save(fig, path)


def plot_edge_case(curve: list[dict], case: dict, model: str, path: Path, title: str = "",
                   pi_label: str = "PI") -> Path:
    """경계 사례 1건: 같은 형상·Re의 CL-α, log10 CD-α.

    CFD(SST) 정답(채움 = converged, 빈 원 = oscillating), 모델 예측과 예측 구간, Transition SST 정답(있으면),
    고른 받음각(회색 점선)과 CFD CL 최대 받음각(초록 점선).
    """
    curve = sorted(curve, key=lambda r: r["aoa"])
    aoa = np.array([r["aoa"] for r in curve])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, target, label in zip(axes, ("cl", "log10_cd"), ("CL", "log10 CD")):
        lo = [r.get(f"{model}_{target}_lo") for r in curve]
        hi = [r.get(f"{model}_{target}_hi") for r in curve]
        if all(v is not None for v in lo + hi):
            ax.fill_between(aoa, np.array(lo, float), np.array(hi, float), color="C0", alpha=0.18, lw=0,
                            label=f"{model} {pi_label}")
        ax.plot(aoa, [r[f"{model}_{target}"] for r in curve], "-", color="C0", lw=1.3, label=f"{model} prediction")
        for state, face in (("converged", "k"), ("oscillating", "none")):
            pts = [(r["aoa"], r[target]) for r in curve if r["conv_state"] == state]
            if pts:
                ax.plot(*zip(*pts), "o", color="k", mfc=face, ms=4.5, label=f"CFD SST ({state})")
        tsst = [(r["aoa"], r[f"{target}_tsst"]) for r in curve if r.get(f"{target}_tsst") is not None]
        if tsst:
            ax.plot(*zip(*tsst), "s", color="C3", mfc="none", ms=6, label="CFD Transition SST")
        ax.axvline(case["aoa"], color="0.45", ls="--", lw=1)
        if case.get("aoa_clmax") is not None:
            ax.axvline(case["aoa_clmax"], color="C2", ls=":", lw=1.2, label="CL max (CFD)")
        ax.set_xlabel("aoa [deg]")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle(title, fontsize=9)
    return _save(fig, path)
