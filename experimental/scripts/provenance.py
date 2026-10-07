#!/usr/bin/env python3
"""[2026-09-19] Provenance stamps on every derived artefact, and a hard abort when a consumer's expectations differ.

The failure this prevents is specific and already happened twice today. `track_export.npz` is written by one
script and appended to by three more, so a file on disk can hold a body position from one canonical-model
setting, a TCP observation from another detector version, and be fitted by a script that assumes a third. Every
one of those states looks identical from the outside -- a valid npz with the right key names -- and the fit that
results is wrong in a way no residual reveals.

So: producers stamp what they were, consumers say what they need, and a mismatch stops the run rather than
flowing into a calibration. The stamps are plain scalars/strings inside the same npz, because provenance kept in
a sidecar file is provenance that gets separated from its data.

    from scripts.provenance import stamp, require
    out.update(stamp(source=ep_path, side=side, canonical_visibility=0.05, pose_balanced=True))
    require(z, canonical_visibility=0.05, pose_balanced=True)
"""
from __future__ import annotations
import subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

PREFIX = "prov_"


def git_hash(root: Path | None = None) -> str:
    root = root or Path(__file__).resolve().parents[1]
    try:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                   capture_output=True, text=True, timeout=5).stdout.strip()
            return r.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        pass
    return "no-git"          # this tree is not a repository; recorded honestly rather than left blank


def stamp(**fields) -> dict:
    """Provenance keys for an npz. Values are coerced to numpy scalars so np.savez keeps them."""
    fields.setdefault("written_utc", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    fields.setdefault("git", git_hash())
    fields.setdefault("tool", Path(sys.argv[0]).name)
    out = {}
    for k, v in fields.items():
        if isinstance(v, Path):
            v = str(v)
        if isinstance(v, bool):
            v = np.bool_(v)
        elif isinstance(v, (int, np.integer)):
            v = np.int64(v)
        elif isinstance(v, float):
            v = np.float64(v)
        else:
            v = np.str_(str(v))
        out[PREFIX + k] = v
    return out


def read(z, key: str, default=None):
    k = PREFIX + key
    if k not in z:
        return default
    v = z[k]
    v = v.item() if getattr(v, "shape", ()) == () else v
    return v.decode() if isinstance(v, bytes) else v


def require(z, *, context: str = "", **expected) -> None:
    """Abort unless every expected stamp matches. Missing stamps are a mismatch, not a pass.

    An artefact written before stamping existed cannot be distinguished from one written with the wrong
    settings, and guessing in the consumer's favour is exactly how a stale file survives.
    """
    bad = []
    for k, want in expected.items():
        got = read(z, k, None)
        if got is None:
            bad.append(f"{k}: NOT STAMPED (expected {want!r}) -- regenerate this artefact")
        elif isinstance(want, float):
            if abs(float(got) - want) > 1e-9:
                bad.append(f"{k}: {got!r} != {want!r}")
        elif bool(got) != bool(want) if isinstance(want, bool) else str(got) != str(want):
            bad.append(f"{k}: {got!r} != {want!r}")
    if bad:
        raise SystemExit("PROVENANCE MISMATCH" + (f" ({context})" if context else "") + ":\n  "
                         + "\n  ".join(bad)
                         + "\n\nThe artefact on disk was not produced by the settings this step assumes. "
                           "Re-run the producing step; do not proceed.")


def describe(z) -> str:
    keys = sorted(k for k in getattr(z, "files", []) if k.startswith(PREFIX))
    if not keys:
        return "  (no provenance stamps -- artefact predates stamping, treat as unknown)"
    return "\n".join(f"  {k[len(PREFIX):]:<26s} {read(z, k[len(PREFIX):])}" for k in keys)


def selftest() -> int:
    import tempfile
    d = Path(tempfile.mkdtemp())
    f = d / "t.npz"
    np.savez(f, x=np.arange(3), **stamp(source="ep1", side="right", canonical_visibility=0.05, pose_balanced=True))
    z = np.load(f)
    print(describe(z))
    require(z, canonical_visibility=0.05, pose_balanced=True, side="right")
    print("\n  matching stamps accepted")
    for kw, why in (({"canonical_visibility": 0.20}, "wrong visibility"),
                    ({"pose_balanced": False}, "wrong balancing"),
                    ({"cube_version": "planes_v1"}, "unstamped key")):
        try:
            require(z, **kw)
        except SystemExit:
            print(f"  rejected: {why}")
        else:
            print(f"  FAIL: {why} was accepted"); return 1
    print("\nSELFTEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(selftest())
