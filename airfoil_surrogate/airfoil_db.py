"""UIUC Airfoil Coordinates Database에서 좌표 파일 내려받기 (afs fetch-airfoils).

케이스 변수 main_airfoil이 좌표 파일 경로(예: "airfoils/s1223.dat")인데 파일이 없으면
파일 이름(s1223)으로 UIUC 사이트에서 받아 그 경로에 저장한다.
사이트 접속이 막혀 있으면 UIUC의 전체 묶음(coord_seligFmt.zip)을 따로 받아 --zip 으로 지정한다.
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

from .cases import Case
from .config import ConfigError, ProjectConfig

UIUC_URL = "https://m-selig.ae.illinois.edu/ads/coord/{name}.dat"
UIUC_ZIP_URL = "https://m-selig.ae.illinois.edu/ads/archives/coord_seligFmt.zip"


def coordinate_files(cfg: ProjectConfig, cases: list[Case]) -> dict[str, Path]:
    """{main_airfoil 값: 좌표 파일 경로} (NACA 4자리 코드는 제외)."""
    out = {}
    for c in cases:
        spec = str(c.params.get("main_airfoil", "")).strip()
        code = spec.upper().removeprefix("NACA").strip()
        if spec and not (len(code) == 4 and code.isdigit()):
            out[spec] = (cfg.root / spec).resolve()
    return out


def _valid(data: bytes) -> bool:
    numeric = 0
    for line in data.decode("utf-8", errors="replace").splitlines():
        parts = line.replace(",", " ").split()
        try:
            float(parts[0]), float(parts[1])
            numeric += 1
        except (IndexError, ValueError):
            continue
    return numeric >= 10


def fetch_airfoils(cfg: ProjectConfig, cases: list[Case], url: str = UIUC_URL,
                   zip_path: Path | None = None, force: bool = False, timeout: float = 30) -> dict:
    """없는 좌표 파일을 내려받는다. 반환: {"ok": [...], "skipped": [...], "failed": {이름: 이유}}."""
    archive = None
    if zip_path is not None:
        if not zip_path.is_file():
            raise ConfigError(f"zip 파일을 찾을 수 없습니다: {zip_path}")
        archive = zipfile.ZipFile(zip_path)
        members = {Path(n).stem.lower(): n for n in archive.namelist() if n.lower().endswith(".dat")}
    result: dict = {"ok": [], "skipped": [], "failed": {}}
    for spec, path in sorted(coordinate_files(cfg, cases).items()):
        if path.is_file() and not force:
            result["skipped"].append(spec)
            continue
        name = path.stem
        try:
            if archive is not None:
                member = members.get(name.lower())
                if member is None:
                    raise FileNotFoundError("zip 안에 없음")
                data = archive.read(member)
            else:
                req = urllib.request.Request(url.format(name=name), headers={"User-Agent": "airfoil-surrogate"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = resp.read()
            if not _valid(data):
                raise ValueError("좌표 형식이 아님")
        except Exception as exc:  # 네트워크, 404, 형식 오류 모두 이름별로 보고
            result["failed"][spec] = str(exc)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        result["ok"].append(spec)
    return result
