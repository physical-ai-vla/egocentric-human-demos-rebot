"""Give recorded data back to the person who recorded it.

The Orbbec SDK cannot claim its UVC interface without root on macOS, so every session that includes the head RGB-D
camera is recorded under sudo -- and everything it writes ends up owned by root. Nothing downstream runs as root, so
`pose_process` fails with PermissionError on its own episode, and so would any later processing, packing or deletion.
That is not a problem to solve once by hand; it recurs on every capture.

`sudo` leaves SUDO_UID and SUDO_GID in the environment precisely so a program can undo this for the files it creates."""
from __future__ import annotations
import logging
import os
from pathlib import Path

log = logging.getLogger("handumi.ownership")


def invoking_user() -> tuple[int, int] | None:
    """(uid, gid) of whoever ran sudo, or None when this is not a privileged run that needs undoing."""
    if os.geteuid() != 0: return None
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if uid is None: return None
    return int(uid), int(gid or 0)


def give_back(path: Path) -> None:
    """Hand `path` (and everything under it) to the invoking user. A no-op when not running under sudo.

    Failures are logged and swallowed: losing the recording because the chown did not work would be a far worse outcome
    than a file the user has to take ownership of later."""
    who = invoking_user()
    if who is None or not path.exists(): return
    uid, gid = who
    try:
        os.chown(path, uid, gid)
        if path.is_dir():
            for p in path.rglob("*"):
                try: os.chown(p, uid, gid)
                except OSError as exc: log.warning("chown %s: %s", p, exc)
    except OSError as exc:
        log.warning("chown %s: %s", path, exc)
