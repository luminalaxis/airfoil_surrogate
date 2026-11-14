"""gmsh 기반 2D 혼합 격자 자동 생성 (벽면 경계층 사각형 + 외부 삼각형).

형상 파라미터가 같은 케이스(받음각만 다른 케이스)는 메쉬를 공유한다.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from .config import ConfigError, safe_name
from .fluent_msh import Boundary, check_msh, write_msh
from .geometry import Geometry, build_geometry, geometry_params

MESH_FILE = "mesh.msh"
INFO_FILE = "mesh_info.json"
ERROR_FILE = "mesh_error.txt"


@dataclass
class MeshSettings:
    chord: float = 1.0            # 메인익 시위 [m] (메쉬 좌표 스케일)
    reynolds: float = 3e6         # 메인익 시위 기준 레이놀즈수 (첫 셀 높이 계산용)
    y_plus: float = 1.0           # 목표 y+
    growth_rate: float = 1.15     # 경계층 성장률
    bl_thickness: float = 0.02    # 경계층 사각형 영역 최대 두께 [c]
    surface_cells: int = 200      # 익형 윗면/아랫면 각각 셀 수
    te_cells: int = 6             # 뭉툭한 뒷전 면 셀 수
    surface_bump: float = 0.15    # 앞전/뒷전 집중도 (작을수록 양 끝에 집중)
    near_size: float = 0.01       # 물체 주변 삼각형 크기 [c]
    wake_size: float = 0.01       # 후류 영역 셀 크기 [c]
    wake_length: float = 3.0      # 후류 세분화 길이 [c]
    wake_angles: list = None      # 뒷전에서 뻗는 후류선 각도 [deg] (받음각 범위를 덮도록)
    # 영역 모양: "circle" = 원형 원거리 경계, "box" = 직사각형 인클로저
    farfield_shape: str = "circle"
    farfield_radius: float = 50.0 # (circle) 원거리 경계 반지름 [c]
    farfield_size: float = 5.0    # (circle) 원거리 경계 셀 크기 [c]
    farfield_split: bool = False  # (circle) true면 앞쪽 반원 inlet(velocity-inlet) +
                                  # 뒤쪽 반원 outlet(pressure-outlet)으로 나눔 (밀도 일정 비압축 해석용)
    farfield_outlet_angle: float = 90.0  # (circle, split) outlet 호의 반각 [deg]. 뒤쪽 ±이 각도만 outlet, 나머지는 inlet.
                                  # 받음각 α에서 outlet으로 공기가 거꾸로 들어오지 않으려면 90 − |α|보다 작게 (예: 60 → |α| < 30°)
    # (box) 직사각형 인클로저: 앞 = inlet(velocity-inlet), 위·아래 = top·bottom(wall,
    #       Fluent에서 미끄럼 벽 = Specified Shear 0), 뒤 = outlet(pressure-outlet)
    box_upstream: float = 20.0    # 앞전 ~ inlet(앞 경계) 거리 [c]
    box_downstream: float = 40.0  # 앞전 ~ outlet(뒤 경계) 거리 [c]
    box_height: float = 20.0      # 시위선 ~ 위쪽/아래쪽 경계 거리 [c] (전체 높이는 2배)
    box_far_size: float = 2.0     # 바깥 경계 셀 크기 [c]
    min_quality: float = 0.05     # 허용 최소 셀 품질 (0~1, 삼각형 SICN / 사각형 SJ)
    refine: float = 1.0           # 메쉬 전체 세분화 비율 (메쉬 독립성 확인용). 1.5면 모든 셀 크기·첫 셀 높이 ÷1.5,
                                  # 표면 셀 수 ×1.5, 경계층 성장률^(1/1.5). 뒷전 세분화도 같은 비율
    timeout: float = 300.0        # 형상 하나의 메쉬 생성 제한 시간 [s]. 넘으면 그 형상만 실패 처리 (0 = 제한 없음)

    @classmethod
    def from_dict(cls, data: dict) -> "MeshSettings":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"[mesh]에 알 수 없는 항목: {', '.join(sorted(unknown))}")
        obj = cls(**data)
        if obj.wake_angles is None:
            obj.wake_angles = [-10.0, 0.0, 10.0, 20.0]
        if obj.farfield_shape not in ("circle", "box"):
            raise ConfigError(f"[mesh] farfield_shape는 'circle' 또는 'box'여야 합니다: {obj.farfield_shape!r}")
        if obj.farfield_shape == "box" and min(obj.box_upstream, obj.box_downstream - 1, obj.box_height) <= 0:
            raise ConfigError("[mesh] box_upstream, box_height는 0보다, box_downstream은 1(시위)보다 커야 합니다.")
        if not obj.refine > 0:
            raise ConfigError(f"[mesh] refine은 0보다 커야 합니다: {obj.refine}")
        return obj

    def first_cell_height(self) -> float:
        """평판 난류 마찰계수 추정으로 목표 y+에 맞는 첫 셀 높이 [c] (refine으로 나눔)."""
        cf = 0.026 / self.reynolds ** (1 / 7)
        return self.y_plus / self.refine * math.sqrt(2 / cf) / self.reynolds

    def bl_ratio(self) -> float:
        """경계층 성장률 (refine을 반영: 같은 두께를 refine배 많은 층으로)."""
        return self.growth_rate ** (1 / self.refine)


_NOT_IN_FINGERPRINT = ("timeout",)          # 메쉬 결과에 영향 없는 항목
_ADDED_DEFAULTS = {                         # 나중에 추가된 항목과 그 기본값
    "farfield_split": False, "farfield_shape": "circle",
    "box_upstream": 20.0, "box_downstream": 40.0, "box_height": 20.0, "box_far_size": 2.0,
    "refine": 1.0, "farfield_outlet_angle": 90.0,
}
# 다요소 날개 도구(AERO_AUTOMATION)에서 갈라져 나오며 없앤 항목과 그 기본값.
# 지문에는 그대로 넣어, 분리 전에 만든 메쉬가 다시 만들어지지 않게 한다.
_REMOVED_DEFAULTS = {
    "gap_bl_fraction": 0.3, "flap_surface_cells": 150,
    "ground_upstream": 5.0, "ground_downstream": 12.0, "ground_height": 6.0, "ground_far_size": 0.5,
}


@dataclass
class MeshJob:
    name: str
    dir: Path
    geometry_params: dict
    settings: MeshSettings

    @property
    def mesh_file(self) -> Path:
        return self.dir / MESH_FILE

    def fingerprint(self) -> str:
        settings = {**_REMOVED_DEFAULTS, **asdict(self.settings)}
        for key in _NOT_IN_FINGERPRINT:
            settings.pop(key, None)
        # 나중에 추가된 항목은 기본값이면 빼서, 기존 메쉬가 불필요하게 다시 만들어지지 않게 한다
        for key, default in _ADDED_DEFAULTS.items():
            if settings.get(key) == default:
                settings.pop(key)
        if settings.get("farfield_shape") == "box":
            # box 위·아래 경계를 inlet → wall(top/bottom)로 바꾼 뒤의 메쉬. 예전 box 메쉬는 다시 만든다
            settings["box_bc"] = "top_bottom_wall"
        payload = json.dumps([self.geometry_params, settings], sort_keys=True, default=str)
        return hashlib.sha1(payload.encode()).hexdigest()[:12]

    def is_current(self) -> bool:
        info = self.dir / INFO_FILE
        if not (self.mesh_file.is_file() and info.is_file()):
            return False
        try:
            return json.loads(info.read_text(encoding="utf-8")).get("fingerprint") == self.fingerprint()
        except json.JSONDecodeError:
            return False


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return Path(str(v)).stem if ("/" in str(v) or "." in str(v)) else str(v)


def mesh_name(params: dict) -> str:
    """메쉬(형상) 이름. 예: 2412, 2412c300, s1223_c0.3. 케이스 변수 mesh_level이 있으면 끝에 _m<단계>
    (같은 형상을 메쉬 설정만 바꿔 푸는 메쉬 독립성 확인용)."""
    p = geometry_params(params)
    name = _fmt(p["main_airfoil"]) + (f"c{_fmt(p['main_chord'])}" if "main_chord" in p else "")
    if "main_chord" not in p and "chord" in p:
        name += f"_c{_fmt(p['chord'])}"
    if params.get("mesh_level") not in (None, ""):
        name += f"_m{_fmt(params['mesh_level'])}"
    return name


def make_job(params: dict, mesh_cfg: dict, meshes_dir: Path) -> MeshJob:
    if "main_airfoil" not in params:
        raise ConfigError("메쉬 생성에는 main_airfoil 변수가 필요합니다 ([constants] 또는 케이스에 지정).")
    geo = geometry_params(params)
    # [mesh] 항목과 같은 이름의 케이스 변수가 있으면 케이스 값 사용 (예: reynolds)
    merged = {**mesh_cfg, **{k: params[k] for k in mesh_cfg.keys() | {f.name for f in fields(MeshSettings)} if k in params}}
    settings = MeshSettings.from_dict(merged)
    name = safe_name(mesh_name(params))
    return MeshJob(name, meshes_dir / name, geo, settings)


# ---------------------------------------------------------------- gmsh 메쉬 생성

def _build_gmsh_model(geom: Geometry, s: MeshSettings, bl_thickness: float):
    import gmsh

    geo = gmsh.model.geo
    walls: dict[str, list[int]] = {}
    fan_points: list[int] = []
    te_points: list[int] = []
    lc = 1.0
    f = s.refine  # 메쉬 독립성 확인용 세분화 비율: 모든 크기 ÷f, 셀 수 ×f
    near_size, wake_size = s.near_size / f, s.wake_size / f

    for el in geom.elements:
        xy = el.xy
        n_surf = max(4, round(s.surface_cells * f))
        sharp = el.sharp_te
        pts = xy[:-1] if sharp else xy
        tags = [geo.addPoint(x, y, 0, lc) for x, y in pts]
        if sharp:
            tags.append(tags[0])
        le = el.le_index
        upper = geo.addSpline(tags[: le + 1])
        lower = geo.addSpline(tags[le:])
        curves = [upper, lower]
        for c in (upper, lower):
            geo.mesh.setTransfiniteCurve(c, n_surf + 1, "Bump", s.surface_bump)
        if sharp:
            fan_points.append(tags[0])
        else:
            te = geo.addLine(tags[-1], tags[0])
            geo.mesh.setTransfiniteCurve(te, max(2, round(s.te_cells * f)) + 1)
            curves.append(te)
            fan_points += [tags[0], tags[-1]]
        te_points += [tags[0], tags[-1]]
        walls[el.name] = curves

    loops = [geo.addCurveLoop(c) for c in walls.values()]

    if s.farfield_shape == "box":
        # 자유류 직사각형 인클로저: 앞 = inlet(velocity-inlet), 위·아래 = top·bottom(wall,
        # Fluent 베이스 케이스에서 미끄럼 벽 = Specified Shear 0으로 설정), 뒤 = outlet(pressure-outlet)
        far_size = s.box_far_size / f
        grow = 0.5 * min(s.box_upstream, s.box_height)
        x0, x1 = -s.box_upstream, s.box_downstream
        y0, y1 = -s.box_height, s.box_height
        c = [geo.addPoint(x, y, 0, lc) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        bottom, right, top, left = (geo.addLine(c[i], c[(i + 1) % 4]) for i in range(4))
        outer_loop = geo.addCurveLoop([bottom, right, top, left])
        outer = {"inlet": ([left], "velocity-inlet"), "top": ([top], "wall"),
                 "bottom": ([bottom], "wall"), "outlet": ([right], "pressure-outlet")}
    else:
        # 자유류: 원형 원거리 경계 하나 (pressure-far-field)
        R = s.farfield_radius
        far_size = s.farfield_size / f
        grow = 0.5 * R  # 이 거리에 걸쳐 셀이 far_size까지 커짐
        cx = 0.5
        center = geo.addPoint(cx, 0, 0, lc)
        # 원 위의 점: 0, β, π, 2π − β (β = outlet 반각, 기본 90° → 0, 90, 180, 270°)
        beta = math.radians(min(max(s.farfield_outlet_angle, 10.0), 170.0))
        ring = [geo.addPoint(cx + R * math.cos(a), R * math.sin(a), 0, lc)
                for a in (0, beta, math.pi, 2 * math.pi - beta)]
        arcs = [geo.addCircleArc(ring[i], center, ring[(i + 1) % 4]) for i in range(4)]
        outer_loop = geo.addCurveLoop(arcs)
        if s.farfield_split:
            # arcs: 0 = 뒤-위(0~β), 1 = 앞-위(β~180°), 2 = 앞-아래(180°~360°−β), 3 = 뒤-아래(360°−β~360°)
            outer = {"inlet": ([arcs[1], arcs[2]], "velocity-inlet"),
                     "outlet": ([arcs[3], arcs[0]], "pressure-outlet")}
        else:
            outer = {"farfield": (arcs, "pressure-far-field")}
    surface = geo.addPlaneSurface([outer_loop, *loops])
    geo.synchronize()

    for name, curves in walls.items():
        gmsh.model.addPhysicalGroup(1, curves, name=name)
    for name, (curves, _) in outer.items():
        gmsh.model.addPhysicalGroup(1, curves, name=name)
    gmsh.model.addPhysicalGroup(2, [surface], name="fluid")

    fld = gmsh.model.mesh.field
    all_walls = [c for cs in walls.values() for c in cs]

    bl = fld.add("BoundaryLayer")
    fld.setNumbers(bl, "CurvesList", all_walls)
    fld.setNumber(bl, "Size", s.first_cell_height())
    fld.setNumber(bl, "Ratio", s.bl_ratio())
    fld.setNumber(bl, "Thickness", bl_thickness)
    fld.setNumber(bl, "Quads", 1)
    fld.setNumbers(bl, "FanPointsList", fan_points)
    fld.setAsBoundaryLayer(bl)

    dist = fld.add("Distance")
    fld.setNumbers(dist, "CurvesList", all_walls)
    fld.setNumber(dist, "Sampling", 400)
    near = fld.add("Threshold")
    fld.setNumber(near, "InField", dist)
    fld.setNumber(near, "SizeMin", near_size)
    fld.setNumber(near, "SizeMax", far_size)
    fld.setNumber(near, "DistMin", 0.05)
    fld.setNumber(near, "DistMax", grow)
    fld.setNumber(near, "Sigmoid", 0)

    # 후류: 각 물체 뒷전에서 여러 각도로 뻗는 선 주변을 세분화 (받음각이 바뀌어도 메쉬 공유)
    wake_pts = []
    for el in geom.elements:
        te = 0.5 * (el.te_upper + el.te_lower)
        for ang in s.wake_angles:
            d = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
            for k in np.arange(0.02, s.wake_length, wake_size * 2):
                x, y = te + k * d
                wake_pts.append(geo.addPoint(x, y, 0, lc))
    geo.synchronize()
    wake_dist = fld.add("Distance")
    fld.setNumbers(wake_dist, "PointsList", wake_pts)
    wake = fld.add("Threshold")
    fld.setNumber(wake, "InField", wake_dist)
    fld.setNumber(wake, "SizeMin", wake_size)
    fld.setNumber(wake, "SizeMax", far_size)
    fld.setNumber(wake, "DistMin", 0.02)
    fld.setNumber(wake, "DistMax", grow)
    fld.setNumber(wake, "Sigmoid", 0)

    # 뒷전: 경계층 부채꼴(fan) 바깥 삼각형이 찌그러지지 않도록 경계층 두께보다 잘게 세분화
    te_all = fld.add("Distance")
    fld.setNumbers(te_all, "PointsList", te_points)
    te_ref = fld.add("Threshold")
    fld.setNumber(te_ref, "InField", te_all)
    fld.setNumber(te_ref, "SizeMin", min(near_size, max(bl_thickness / 5, 1e-4) / f))
    fld.setNumber(te_ref, "SizeMax", far_size)
    fld.setNumber(te_ref, "DistMin", 3 * bl_thickness)
    fld.setNumber(te_ref, "DistMax", grow)

    size_fields = [near, wake, te_ref]
    bg = fld.add("Min")
    fld.setNumbers(bg, "FieldsList", size_fields)
    fld.setAsBackgroundMesh(bg)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    return walls, outer


def _extract(walls: dict, outer: dict) -> tuple[np.ndarray, list, list[Boundary], float]:
    import gmsh

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    xyz = coords.reshape(-1, 3)
    index = {int(t): i for i, t in enumerate(node_tags)}

    cells, q_min = [], 1.0
    etypes, etags, enodes = gmsh.model.mesh.getElements(2)
    for et, tags, nodes in zip(etypes, etags, enodes):
        # 삼각형: 형상 품질(SICN), 사각형: 뒤틀림(scaled Jacobian).
        # 경계층 사각형은 종횡비가 커서 SICN이 낮게 나오는 게 정상이므로 SJ로 평가한다.
        if et == 2:
            nn, metric = 3, "minSICN"
        elif et == 3:
            nn, metric = 4, "minSJ"
        else:
            raise ConfigError(f"지원하지 않는 2D 요소 타입: {et}")
        nodes = nodes.reshape(-1, nn)
        cells += [tuple(index[int(n)] for n in row) for row in nodes]
        q_min = min(q_min, float(np.min(gmsh.model.mesh.getElementQualities(tags, metric))))

    used = sorted({n for c in cells for n in c})
    remap = {old: new for new, old in enumerate(used)}
    nodes_xy = xyz[used, :2]
    cells = [tuple(remap[n] for n in c) for c in cells]

    boundaries = []
    named = [(n, c, "wall") for n, c in walls.items()] + [(n, c, bc) for n, (c, bc) in outer.items()]
    for name, curves, bc in named:
        edges = []
        for c in curves:
            _, _, enodes1 = gmsh.model.mesh.getElements(1, c)
            for row in enodes1[0].reshape(-1, 2):
                edges.append((remap[index[int(row[0])]], remap[index[int(row[1])]]))
        boundaries.append(Boundary(name, bc, edges))
    return nodes_xy, cells, boundaries, q_min


class MeshWorker:
    """메쉬 생성을 별도 프로세스에서 실행한다 (gmsh가 멈추는 형상이 있어도 시간 제한 후 그 형상만 실패).

    형상마다 새 프로세스에서 만든다: gmsh는 한 프로세스 안에서 앞서 만든 메쉬에 따라(메모리 배치 등)
    같은 설정이어도 셀이 조금씩 달라지는데, 새 프로세스의 첫 메쉬는 항상 같다. 그래서 만드는 순서·동시 개수·
    팀원 분담(--part)과 관계없이 같은 메쉬가 나온다. 시간 초과가 나면 그 프로세스를 끝내고 새로 띄운다.
    """

    def __init__(self):
        self._pool = None

    def __enter__(self) -> "MeshWorker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _get_pool(self):
        if self._pool is None:
            import multiprocessing as mp
            self._pool = mp.get_context("spawn").Pool(1, maxtasksperchild=1)
        return self._pool

    def close(self, kill: bool = False) -> None:
        if self._pool is not None:
            if kill:
                self._pool.terminate()
            else:
                self._pool.close()
            self._pool.join()
            self._pool = None

    def run(self, job: MeshJob, root: Path, plots: bool, stop=None) -> dict:
        """작업 프로세스에서 메쉬 생성. stop(threading.Event)이 켜지면 바로 프로세스를 끝내고 중단한다."""
        timeout = job.settings.timeout if job.settings.timeout and job.settings.timeout > 0 else None
        deadline = None if timeout is None else time.monotonic() + timeout
        result = self._get_pool().apply_async(_generate, (job, root, plots))
        while not result.ready():
            if stop is not None and stop.is_set():
                self.close(kill=True)
                raise ConfigError(f"중단됨 ({job.name})")
            if deadline is not None and time.monotonic() >= deadline:
                self.close(kill=True)
                raise ConfigError(f"메쉬 생성 시간 초과 ({job.settings.timeout:g}초, {job.name}). "
                                  "형상 좌표(뒷전이 겹치거나 꼬인 곳)를 확인하세요.")
            result.wait(0.5)
        return result.get()


_BLAS_THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


@contextmanager
def single_thread_blas():
    """이 안에서 새로 띄우는(spawn) 작업 프로세스는 행렬 계산(BLAS) 스레드를 1개만 쓴다.

    프로세스 여러 개가 각자 코어 수만큼 스레드를 띄우면 서로 경합해 수십 배 느려진다.
    spawn으로 띄운 프로세스는 numpy를 불러오기 전에 이 환경변수를 읽는다.
    """
    saved = {k: os.environ.get(k) for k in _BLAS_THREAD_VARS}
    os.environ.update({k: "1" for k in _BLAS_THREAD_VARS})
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def generate_mesh(job: MeshJob, root: Path, plots: bool = True, worker: MeshWorker | None = None,
                  stop=None) -> dict:
    """메쉬 생성 → mesh.msh, mesh_info.json, 이미지 저장. 실패 시 ConfigError.

    worker를 주면 그 작업 프로세스에서 만든다(재사용). 없으면 settings.timeout > 0 일 때만 별도
    프로세스에서 시간 제한을 두고 실행한다. 예상 못 한 오류도 ConfigError로 바꿔 그 형상만 실패 처리한다.
    """
    job.dir.mkdir(parents=True, exist_ok=True)
    # 재생성이 실패했을 때 이전 설정의 메쉬가 쓰이지 않도록 먼저 지운다
    for name in (MESH_FILE, INFO_FILE, ERROR_FILE):
        (job.dir / name).unlink(missing_ok=True)
    try:
        if worker is not None:
            info = worker.run(job, root, plots, stop)
        elif not job.settings.timeout or job.settings.timeout <= 0:
            info = _generate(job, root, plots)
        else:
            with MeshWorker() as w:
                info = w.run(job, root, plots)
    except ConfigError as exc:
        (job.dir / ERROR_FILE).write_text(str(exc), encoding="utf-8")
        raise
    except Exception as exc:  # gmsh 밖(메쉬 파일 쓰기·검증 등)에서 난 오류도 그 형상만 실패
        (job.dir / MESH_FILE).unlink(missing_ok=True)
        msg = f"메쉬 생성 오류 ({job.name}): {type(exc).__name__}: {exc}"
        (job.dir / ERROR_FILE).write_text(msg, encoding="utf-8")
        raise ConfigError(msg) from exc
    return info


def _generate(job: MeshJob, root: Path, plots: bool) -> dict:
    import gmsh

    s = job.settings
    geom = build_geometry(job.geometry_params, root)

    max_bl = s.bl_thickness

    errors = []
    for attempt, factor in enumerate((1.0, 0.6, 0.35), start=1):
        bl_thickness = max_bl * factor
        gmsh.initialize(interruptible=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add(job.name)
            walls, outer = _build_gmsh_model(geom, s, bl_thickness)
            gmsh.model.mesh.generate(2)
            nodes, cells, boundaries, q_min = _extract(walls, outer)
        except Exception as exc:  # gmsh 내부 오류
            errors.append(f"시도 {attempt}: {exc}")
            continue
        finally:
            gmsh.finalize()
        if q_min < s.min_quality:
            errors.append(f"시도 {attempt}: 최소 셀 품질 {q_min:.3f} < {s.min_quality}")
            continue
        break
    else:
        raise ConfigError(f"메쉬 생성 실패 ({job.name}): " + " / ".join(errors))

    nodes = nodes * s.chord
    stats = write_msh(job.mesh_file, nodes, cells, boundaries)
    check = check_msh(job.mesh_file)
    if check["bad_cells"] or check["open_cells"]:
        raise ConfigError(f"Fluent 메쉬 검증 실패 ({job.name}): {check}")

    h1 = s.first_cell_height()
    info = {
        "name": job.name,
        "fingerprint": job.fingerprint(),
        "geometry": job.geometry_params,
        "geometry_info": geom.info,
        "settings": asdict(s),
        "first_cell_height_c": h1,
        "first_cell_height_m": h1 * s.chord,
        "bl_thickness_c": bl_thickness,
        "bl_layers": int(math.log(1 + bl_thickness * (s.bl_ratio() - 1) / h1) / math.log(s.bl_ratio())),
        "min_quality": q_min,
        "min_orthogonal_quality": check["min_orthogonal_quality"],  # Fluent Mesh > Quality 와 같은 정의
        "cells_oq_below_0.1": check["cells_oq_below_0.1"],
        "attempts": attempt,
        **stats,
    }
    (job.dir / INFO_FILE).write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    if plots:
        from .plots import plot_geometry, plot_mesh
        plot_geometry(geom, job.dir / "geometry.png", title=job.name)
        plot_mesh(nodes / s.chord, cells, geom, job.dir, title=job.name)
    return info
