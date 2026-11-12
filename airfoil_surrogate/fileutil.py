"""파일 쓰기·잠금 도우미.

write_text_atomic : 임시 파일에 쓴 뒤 바꿔 넣는다. 쓰는 도중에 프로세스가 죽어도 이전 내용 또는 새 내용만 남는다.
FileLock          : 같은 케이스·메쉬를 두 프로세스(afs run을 두 번 띄운 경우 등)가 동시에 다루지 못하게 한다.
                    잠금 파일에 호스트 이름과 PID를 남기고, 같은 컴퓨터에서 그 프로세스가 끝났으면 오래된 잠금으로 보고 지운다.
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

IS_WINDOWS = os.name == "nt"


def write_text_atomic(path: Path, text: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding=encoding)
    # Windows: 다른 프로세스가 대상 파일을 잠깐 열고 있으면(afs status 등) os.replace가 실패할 수 있어 몇 번 재시도
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 19:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.05)


def write_json_atomic(path: Path, data) -> None:
    write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2))


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if IS_WINDOWS:
        # os.kill(pid, 0)은 Windows에서 프로세스를 종료시키므로 쓰면 안 된다
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class LockedError(RuntimeError):
    pass


class FileLock:
    """with FileLock(path): ...  이미 다른 살아 있는 프로세스가 잡고 있으면 LockedError."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._held = False

    def _owner(self) -> dict | None:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        me = {"host": socket.gethostname(), "pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        for _ in range(3):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                owner = self._owner()
                if owner is None:
                    # 다른 프로세스가 방금 만들고 아직 내용을 쓰지 않았을 수 있다
                    time.sleep(0.1)
                    owner = self._owner()
                if owner is not None and owner.get("host") == me["host"] and not pid_alive(int(owner.get("pid", 0))):
                    self.path.unlink(missing_ok=True)  # 같은 컴퓨터에서 끝난(비정상 종료된) 프로세스의 잠금
                    continue
                who = f"{owner.get('host')} PID {owner.get('pid')} ({owner.get('started')})" if owner else "알 수 없음"
                raise LockedError(f"다른 프로세스가 사용 중: {who}. 그 프로세스가 없다면 {self.path} 을 지우세요.")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(me, f)
            self._held = True
            return
        raise LockedError(f"잠금 파일을 만들 수 없습니다: {self.path}")

    def release(self) -> None:
        if self._held:
            self.path.unlink(missing_ok=True)
            self._held = False

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
