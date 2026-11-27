"""Supervise Kairos and the locally installed NapCat on WSL2."""

import argparse
import fcntl
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import ExitStack, suppress
from pathlib import Path
from tempfile import TemporaryDirectory

_CONFIG = Path("config/global.yaml")
_BOTS = Path("config/bots")
_POLL_INTERVAL = 0.1
_NAPCAT_UNIT = "kairos-napcat.service"
_NAPCAT_ENV = {
    "DBUS_SESSION_BUS_ADDRESS",
    "HOME",
    "LANG",
    "LD_LIBRARY_PATH",
    "LOGNAME",
    "PATH",
    "SHELL",
    "TZ",
    "USER",
}


def _require_postgres_socket(dsn: str) -> None:
    from sqlalchemy.engine import make_url
    from sqlalchemy.exc import ArgumentError

    try:
        url = make_url(dsn)
    except ArgumentError:
        raise ValueError("postgres.dsn must be a PostgreSQL connection URL") from None
    hosts = url.normalized_query.get("host", ())
    if (
        url.drivername not in {"postgresql", "postgres"}
        or url.host is not None
        or len(hosts) != 1
        or not hosts[0].startswith("/")
        or "," in hosts[0]
    ):
        raise ValueError(
            "Use a PostgreSQL Unix socket. Set postgres.dsn to a URL such as "
            "postgresql:///kairos?host=/var/run/postgresql"
        )


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def _stop_group(process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 5
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(_POLL_INTERVAL)
    else:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def _try_lock_file(stack: ExitStack, path: Path) -> bool:
    lock = stack.enter_context(path.open("rb"))
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return False
    return True


def _spawn(
    stack: ExitStack,
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.Popen[bytes]:
    process = subprocess.Popen(args, cwd=cwd, env=env, start_new_session=True)
    stack.callback(_stop_group, process)
    return process


def _napcat_qq() -> Path:
    default = Path.home() / "Napcat/opt/QQ/qq"
    qq = Path(os.environ.get("NAPCAT_QQ", str(default))).expanduser().resolve()
    if not qq.is_file() or not os.access(qq, os.X_OK):
        raise ValueError(f"Set NAPCAT_QQ to the executable of the installed NapCat QQ: {qq}")
    return qq


def _napcat_running(qq: Path) -> bool:
    for process in Path("/proc").iterdir():
        if process.name.isdecimal():
            with suppress(OSError):
                if (process / "exe").readlink() == qq:
                    return True
    return False


def _napcat_background_active() -> bool:
    with suppress(FileNotFoundError):
        status = subprocess.run(
            ["systemctl", "--user", "show", "--property=ActiveState", "--value", _NAPCAT_UNIT],
            capture_output=True,
            text=True,
            check=False,
        )
        return status.stdout.strip() in {"active", "activating"}
    return False


def _start_napcat_background() -> int:
    if _napcat_background_active():
        print("NapCat background service is already started.")
        return 0
    qq = _napcat_qq()
    if _napcat_running(qq):
        raise RuntimeError(
            "Stop the running NapCat instance before starting the background service."
        )
    return subprocess.run(
        [
            "systemd-run",
            "--user",
            f"--unit={_NAPCAT_UNIT}",
            "--service-type=exec",
            "--collect",
            "--property=Restart=no",
            "--property=KillMode=mixed",
            "--property=TimeoutStopSec=20s",
            f"--setenv=NAPCAT_QQ={qq}",
            sys.executable,
            str(Path(__file__).resolve()),
            "napcat",
        ],
        check=False,
    ).returncode


def _start_napcat(stack: ExitStack, qq: Path) -> subprocess.Popen[bytes]:
    napcat_env = {
        key: value
        for key, value in os.environ.items()
        if key in _NAPCAT_ENV or key.startswith(("LC_", "XDG_"))
    }
    napcat_env["TMPDIR"] = stack.enter_context(TemporaryDirectory(prefix="kairos-napcat-"))
    return _spawn(stack, ["xvfb-run", "-a", str(qq), "--no-sandbox"], cwd=qq.parent, env=napcat_env)


def _start_kairos(stack: ExitStack, stopping: Callable[[], bool]) -> subprocess.Popen[bytes] | None:
    from kairos.config import load_bot_profiles, load_global_config

    if not _try_lock_file(stack, _CONFIG):
        raise RuntimeError("Kairos is already running from this launcher")
    config = load_global_config(_CONFIG)
    if not 1 <= len(load_bot_profiles(_BOTS, config)) <= 3:
        raise ValueError("Configure between 1 and 3 active bot profiles in config/bots")
    _require_postgres_socket(config.postgres.dsn)
    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(config.onebot_ws.host, config.onebot_ws.host)
    port = config.onebot_ws.port
    if port == 0 or _port_is_open(host, port):
        raise ValueError("Choose an available, fixed onebot_ws.port for NapCat")

    with ExitStack() as migration_scope:
        migration = _spawn(
            migration_scope,
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            env={**os.environ, "KAIROS_PG_DSN": config.postgres.dsn},
        )
        while not stopping() and migration.poll() is None:
            time.sleep(_POLL_INTERVAL)
    if stopping():
        return None
    if migration.returncode:
        raise RuntimeError("PostgreSQL migration failed")

    kairos = _spawn(stack, [sys.executable, "-m", "kairos.main"])
    deadline = time.monotonic() + 30
    while not stopping():
        if kairos.poll() is not None:
            raise RuntimeError("Kairos exited before the OneBot listener was ready")
        if _port_is_open(host, port):
            return kairos
        if time.monotonic() >= deadline:
            raise RuntimeError("Timed out waiting for the Kairos OneBot listener")
        time.sleep(_POLL_INTERVAL)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "service", nargs="?", choices=("all", "kairos", "napcat", "napcat-start"), default="all"
    )
    mode = parser.parse_args().service
    interrupted = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True

    try:
        if mode == "napcat-start":
            return _start_napcat_background()

        with ExitStack() as stack:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous = signal.signal(signum, request_stop)
                stack.callback(signal.signal, signum, previous)

            qq = _napcat_qq() if mode != "kairos" else None
            services: dict[str, subprocess.Popen[bytes]] = {}
            if mode != "napcat":
                kairos = _start_kairos(stack, lambda: interrupted)
                if kairos is None:
                    return 0
                services["Kairos"] = kairos
            if qq is not None and not interrupted:
                if (
                    _napcat_running(qq)
                    or (mode == "all" and _napcat_background_active())
                    or not _try_lock_file(stack, qq)
                ):
                    print("NapCat is already running.", flush=True)
                    if mode == "napcat":
                        return 0
                else:
                    services["NapCat"] = _start_napcat(stack, qq)

            while not interrupted:
                for name, service in services.items():
                    if (code := service.poll()) is not None:
                        print(f"{name} exited with status {code}.", file=sys.stderr)
                        return 128 - code if code < 0 else code
                time.sleep(_POLL_INTERVAL)
    except (OSError, RuntimeError, ValueError) as exc:
        target = "dev" if mode == "all" else mode
        print(f"make {target}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
