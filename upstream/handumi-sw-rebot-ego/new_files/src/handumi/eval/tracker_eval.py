"""Phase-2 EEF tracker acceptance tests for Quest-tracked HandUMI tips.

Everything here is pure numpy over a *capture CSV* so it can be unit-tested
and re-run offline. The CSV is produced by ``handumi tracking eval capture``
(see :mod:`handumi.scripts.tracker_eval`) and holds one row per Quest sample::

    t_ns, mark, streaming, clock_synced,
    left_tracked, right_tracked,
    lc_x..lc_qw   (left  controller pose7 in the recording/table frame)
    rc_x..rc_qw   (right controller pose7)
    lt_x..lt_qw   (left  TCP pose7 = controller @ T_controller_tcp)
    rt_x..rt_qw   (right TCP pose7)

``mark`` is 0 for free-running rows and ``k`` for rows captured inside the
k-th operator mark window (Enter pressed while the tip is held on a point).

Tests
-----
* :func:`static_jitter` — tip fixed on the table, ~30 s: per-axis sigma and
  translation/rotation RMS. Gate: translation RMS < 5 mm, rotation RMS < 1 deg.
* :func:`return_to_point` — the operator returns to point A ``N`` times and
  marks each visit: ``||p_A(k) - p_A(0)||``. Gate: max < 10 mm.
* :func:`left_right_agreement` — both tips touch the same physical point (one
  mark per touch, or alternating L/R marks): ``||p_L - p_R||``. Gate: max
  < 10 mm (upstream HandUMI fails at 15 mm).
* :func:`table_z` — with a session (table) calibration active every marked
  tip should sit at ``z ~= 0``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

Status = Literal["PASS", "WARN", "FAIL"]
SIDES = ("left", "right")
POSE_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw")
COLUMN_PREFIX = {
    ("left", "controller"): "lc",
    ("right", "controller"): "rc",
    ("left", "tcp"): "lt",
    ("right", "tcp"): "rt",
}
CSV_HEADER: tuple[str, ...] = (
    "t_ns",
    "mark",
    "streaming",
    "clock_synced",
    "left_tracked",
    "right_tracked",
    *[f"{p}_{n}" for p in ("lc", "rc", "lt", "rt") for n in POSE_NAMES],
)


@dataclass(frozen=True)
class Thresholds:
    static_translation_rms_mm: float = 5.0
    static_rotation_rms_deg: float = 1.0
    return_to_point_max_mm: float = 10.0
    left_right_agreement_max_mm: float = 10.0
    table_z_max_mm: float = 15.0
    # WARN band multiplier: value in [gate, gate*warn_factor) -> WARN, above -> FAIL.
    warn_factor: float = 2.0

    @classmethod
    def from_rig(cls, rig: dict | None) -> "Thresholds":
        values = dict((rig or {}).get("tracker_eval", {}) or {})
        known = {k: float(v) for k, v in values.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def grade(value: float, gate: float, *, warn_factor: float = 2.0) -> Status:
    if not np.isfinite(value):
        return "FAIL"
    if value < gate:
        return "PASS"
    if value < gate * warn_factor:
        return "WARN"
    return "FAIL"


def worst(*statuses: Status) -> Status:
    order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    return max(statuses, key=lambda s: order[s]) if statuses else "PASS"


# ---------------------------------------------------------------------------
# Capture container
# ---------------------------------------------------------------------------


@dataclass
class TrackerCapture:
    t_ns: np.ndarray  # (N,) int64
    mark: np.ndarray  # (N,) int64
    streaming: np.ndarray  # (N,) bool
    clock_synced: np.ndarray  # (N,) bool
    tracked: dict[str, np.ndarray]  # side -> (N,) bool
    controller: dict[str, np.ndarray]  # side -> (N, 7)
    tcp: dict[str, np.ndarray]  # side -> (N, 7)

    def __len__(self) -> int:
        return int(len(self.t_ns))

    @property
    def duration_s(self) -> float:
        if len(self) < 2:
            return 0.0
        return float(self.t_ns[-1] - self.t_ns[0]) * 1e-9

    @property
    def sample_rate_hz(self) -> float:
        d = self.duration_s
        return (len(self) - 1) / d if d > 0 else 0.0

    def poses(self, side: str, kind: str = "tcp") -> np.ndarray:
        return self.tcp[side] if kind == "tcp" else self.controller[side]

    def mark_ids(self) -> list[int]:
        ids = sorted(int(m) for m in np.unique(self.mark) if m > 0)
        return ids

    def mark_pose(
        self, mark_id: int, side: str, kind: str = "tcp", *, tracked_only: bool = True
    ) -> np.ndarray | None:
        """Robust (median position, mean rotation) pose7 over one mark window."""
        rows = self.mark == mark_id
        if tracked_only:
            rows &= self.tracked[side]
        if not rows.any():
            return None
        return robust_mean_pose(self.poses(side, kind)[rows])

    # -- I/O ----------------------------------------------------------------

    def to_rows(self) -> list[list]:
        out: list[list] = []
        for i in range(len(self)):
            row: list = [
                int(self.t_ns[i]),
                int(self.mark[i]),
                int(self.streaming[i]),
                int(self.clock_synced[i]),
                int(self.tracked["left"][i]),
                int(self.tracked["right"][i]),
            ]
            for side, kind in (("left", "controller"), ("right", "controller"), ("left", "tcp"), ("right", "tcp")):
                row.extend(float(v) for v in self.poses(side, kind)[i])
            out.append(row)
        return out

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(CSV_HEADER)
            w.writerows(self.to_rows())

    @classmethod
    def from_arrays(cls, rows: np.ndarray) -> "TrackerCapture":
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != len(CSV_HEADER):
            raise ValueError(f"expected (N, {len(CSV_HEADER)}) rows, got {rows.shape}")
        col = {name: i for i, name in enumerate(CSV_HEADER)}

        def pose(prefix: str) -> np.ndarray:
            return rows[:, [col[f"{prefix}_{n}"] for n in POSE_NAMES]].astype(np.float64)

        return cls(
            t_ns=rows[:, col["t_ns"]].astype(np.int64),
            mark=rows[:, col["mark"]].astype(np.int64),
            streaming=rows[:, col["streaming"]] > 0.5,
            clock_synced=rows[:, col["clock_synced"]] > 0.5,
            tracked={s: rows[:, col[f"{s}_tracked"]] > 0.5 for s in SIDES},
            controller={"left": pose("lc"), "right": pose("rc")},
            tcp={"left": pose("lt"), "right": pose("rt")},
        )

    @classmethod
    def read_csv(cls, path: Path) -> "TrackerCapture":
        with Path(path).open(newline="") as f:
            reader = csv.reader(f)
            header = tuple(next(reader))
            if header != CSV_HEADER:
                raise ValueError(f"{path}: unexpected header {header[:6]}...")
            rows = np.array([[float(v) for v in r] for r in reader if r], dtype=np.float64)
        if rows.size == 0:
            rows = rows.reshape(0, len(CSV_HEADER))
        return cls.from_arrays(rows)


class CaptureBuilder:
    """Incrementally accumulate samples during a live capture."""

    def __init__(self) -> None:
        self._rows: list[list[float]] = []

    def add(
        self,
        *,
        t_ns: int,
        mark: int,
        streaming: bool,
        clock_synced: bool,
        left_tracked: bool,
        right_tracked: bool,
        left_controller: np.ndarray,
        right_controller: np.ndarray,
        left_tcp: np.ndarray,
        right_tcp: np.ndarray,
    ) -> None:
        row = [
            float(t_ns),
            float(mark),
            float(streaming),
            float(clock_synced),
            float(left_tracked),
            float(right_tracked),
        ]
        for pose in (left_controller, right_controller, left_tcp, right_tcp):
            row.extend(float(v) for v in np.asarray(pose, dtype=np.float64).reshape(7))
        self._rows.append(row)

    def __len__(self) -> int:
        return len(self._rows)

    def build(self) -> TrackerCapture:
        rows = np.asarray(self._rows, dtype=np.float64)
        if rows.size == 0:
            rows = rows.reshape(0, len(CSV_HEADER))
        return TrackerCapture.from_arrays(rows)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _canonical_quats(q: np.ndarray) -> np.ndarray:
    """Flip quaternion signs so they lie on one hemisphere relative to the first."""
    q = np.asarray(q, dtype=np.float64).reshape(-1, 4).copy()
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    q /= norms
    ref = q[0]
    flip = np.sum(q * ref[None, :], axis=1) < 0  # (Accelerate matmul emits spurious FP warnings)
    q[flip] *= -1.0
    return q


def mean_rotation(q_xyzw: np.ndarray) -> Rotation:
    q = _canonical_quats(q_xyzw)
    return Rotation.from_quat(q).mean()


def rotation_residuals_deg(q_xyzw: np.ndarray) -> np.ndarray:
    """Per-sample rotation vector (deg, 3 axes) relative to the mean rotation."""
    q = _canonical_quats(q_xyzw)
    rots = Rotation.from_quat(q)
    mean = rots.mean()
    rel = mean.inv() * rots
    return np.degrees(rel.as_rotvec())


def robust_mean_pose(poses7: np.ndarray) -> np.ndarray:
    poses7 = np.asarray(poses7, dtype=np.float64).reshape(-1, 7)
    position = np.median(poses7[:, :3], axis=0)
    quat = mean_rotation(poses7[:, 3:7]).as_quat()
    return np.concatenate([position, quat])


# ---------------------------------------------------------------------------
# Test 1: static jitter
# ---------------------------------------------------------------------------


@dataclass
class SideJitter:
    side: str
    samples: int
    tracked_fraction: float
    sigma_xyz_mm: list[float]
    translation_rms_mm: float
    translation_p95_mm: float
    sigma_rpy_deg: list[float]
    rotation_rms_deg: float
    status: Status


@dataclass
class JitterReport:
    kind: str
    duration_s: float
    sample_rate_hz: float
    sides: dict[str, SideJitter]
    status: Status

    def to_dict(self) -> dict:
        return {
            "test": "static_jitter",
            "kind": self.kind,
            "duration_s": self.duration_s,
            "sample_rate_hz": self.sample_rate_hz,
            "sides": {k: asdict(v) for k, v in self.sides.items()},
            "status": self.status,
        }


def static_jitter(
    capture: TrackerCapture,
    *,
    kind: str = "tcp",
    thresholds: Thresholds = Thresholds(),
    sides: tuple[str, ...] = SIDES,
    min_samples: int = 30,
) -> JitterReport:
    reports: dict[str, SideJitter] = {}
    for side in sides:
        tracked = capture.tracked[side]
        poses = capture.poses(side, kind)[tracked]
        n = int(len(poses))
        fraction = float(tracked.mean()) if len(capture) else 0.0
        if n < min_samples:
            reports[side] = SideJitter(
                side, n, fraction, [float("nan")] * 3, float("nan"), float("nan"),
                [float("nan")] * 3, float("nan"), "FAIL",
            )
            continue
        pos = poses[:, :3]
        centred = pos - np.median(pos, axis=0)
        sigma_xyz = centred.std(axis=0) * 1000.0
        dist_mm = np.linalg.norm(centred, axis=1) * 1000.0
        trans_rms = float(np.sqrt(np.mean(dist_mm**2)))
        trans_p95 = float(np.percentile(dist_mm, 95))
        rot_res = rotation_residuals_deg(poses[:, 3:7])
        sigma_rpy = rot_res.std(axis=0)
        rot_rms = float(np.sqrt(np.mean(np.sum(rot_res**2, axis=1))))
        status = worst(
            grade(trans_rms, thresholds.static_translation_rms_mm, warn_factor=thresholds.warn_factor),
            grade(rot_rms, thresholds.static_rotation_rms_deg, warn_factor=thresholds.warn_factor),
            "PASS" if fraction >= 0.99 else ("WARN" if fraction >= 0.95 else "FAIL"),
        )
        reports[side] = SideJitter(
            side=side,
            samples=n,
            tracked_fraction=fraction,
            sigma_xyz_mm=[float(v) for v in sigma_xyz],
            translation_rms_mm=trans_rms,
            translation_p95_mm=trans_p95,
            sigma_rpy_deg=[float(v) for v in sigma_rpy],
            rotation_rms_deg=rot_rms,
            status=status,
        )
    return JitterReport(
        kind=kind,
        duration_s=capture.duration_s,
        sample_rate_hz=capture.sample_rate_hz,
        sides=reports,
        status=worst(*(r.status for r in reports.values())),
    )


# ---------------------------------------------------------------------------
# Test 2: return-to-point
# ---------------------------------------------------------------------------


@dataclass
class ReturnReport:
    side: str
    kind: str
    reference_mark: int
    reference_position_m: list[float]
    errors_mm: list[float]
    marks: list[int]
    mean_mm: float
    max_mm: float
    status: Status
    skipped_marks: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["test"] = "return_to_point"
        return d


def return_to_point(
    capture: TrackerCapture,
    *,
    side: str,
    kind: str = "tcp",
    thresholds: Thresholds = Thresholds(),
) -> ReturnReport:
    ids = capture.mark_ids()
    poses: list[tuple[int, np.ndarray]] = []
    skipped: list[int] = []
    for m in ids:
        p = capture.mark_pose(m, side, kind)
        (poses if p is not None else skipped).append((m, p) if p is not None else m)
    if len(poses) < 2:
        return ReturnReport(
            side, kind, poses[0][0] if poses else -1,
            [float(v) for v in poses[0][1][:3]] if poses else [],
            [], [], float("nan"), float("nan"), "FAIL", skipped,
        )
    ref_mark, ref_pose = poses[0]
    errors = [float(np.linalg.norm(p[:3] - ref_pose[:3]) * 1000.0) for _, p in poses[1:]]
    marks = [m for m, _ in poses[1:]]
    mx = max(errors)
    return ReturnReport(
        side=side,
        kind=kind,
        reference_mark=ref_mark,
        reference_position_m=[float(v) for v in ref_pose[:3]],
        errors_mm=errors,
        marks=marks,
        mean_mm=float(np.mean(errors)),
        max_mm=mx,
        status=grade(mx, thresholds.return_to_point_max_mm, warn_factor=thresholds.warn_factor),
        skipped_marks=skipped,
    )


# ---------------------------------------------------------------------------
# Test 3: left/right agreement
# ---------------------------------------------------------------------------


@dataclass
class AgreementReport:
    mode: str
    kind: str
    pairs: list[dict]
    separations_mm: list[float]
    mean_mm: float
    max_mm: float
    status: Status

    def to_dict(self) -> dict:
        d = asdict(self)
        d["test"] = "left_right_agreement"
        return d


def left_right_agreement(
    capture: TrackerCapture,
    *,
    mode: Literal["simultaneous", "sequential"] = "simultaneous",
    kind: str = "tcp",
    thresholds: Thresholds = Thresholds(),
) -> AgreementReport:
    """Tip separation when both tips touch one physical point.

    ``simultaneous``: each mark window contains both tips touching (tip-to-tip
    or both on one dimple); separation = ``||p_L - p_R||`` within the window.
    ``sequential``: marks alternate left (odd) / right (even) on the same point.
    """
    ids = capture.mark_ids()
    pairs: list[dict] = []
    seps: list[float] = []
    if mode == "simultaneous":
        for m in ids:
            pl = capture.mark_pose(m, "left", kind)
            pr = capture.mark_pose(m, "right", kind)
            if pl is None or pr is None:
                pairs.append({"mark": m, "skipped": True})
                continue
            sep = float(np.linalg.norm(pl[:3] - pr[:3]) * 1000.0)
            pairs.append({"mark": m, "left_m": pl[:3].tolist(), "right_m": pr[:3].tolist(), "separation_mm": sep})
            seps.append(sep)
    elif mode == "sequential":
        for ml, mr in zip(ids[0::2], ids[1::2]):
            pl = capture.mark_pose(ml, "left", kind)
            pr = capture.mark_pose(mr, "right", kind)
            if pl is None or pr is None:
                pairs.append({"marks": [ml, mr], "skipped": True})
                continue
            sep = float(np.linalg.norm(pl[:3] - pr[:3]) * 1000.0)
            pairs.append({"marks": [ml, mr], "left_m": pl[:3].tolist(), "right_m": pr[:3].tolist(), "separation_mm": sep})
            seps.append(sep)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    if not seps:
        return AgreementReport(mode, kind, pairs, [], float("nan"), float("nan"), "FAIL")
    mx = max(seps)
    return AgreementReport(
        mode=mode,
        kind=kind,
        pairs=pairs,
        separations_mm=seps,
        mean_mm=float(np.mean(seps)),
        max_mm=mx,
        status=grade(mx, thresholds.left_right_agreement_max_mm, warn_factor=thresholds.warn_factor),
    )


# ---------------------------------------------------------------------------
# Test 4: table z (requires session calibration)
# ---------------------------------------------------------------------------


@dataclass
class TableZReport:
    kind: str
    z_mm: dict[str, list[float]]
    max_abs_mm: float
    status: Status

    def to_dict(self) -> dict:
        d = asdict(self)
        d["test"] = "table_z"
        return d


def table_z(
    capture: TrackerCapture,
    *,
    kind: str = "tcp",
    thresholds: Thresholds = Thresholds(),
    sides: tuple[str, ...] = SIDES,
) -> TableZReport:
    z_mm: dict[str, list[float]] = {}
    for side in sides:
        vals: list[float] = []
        for m in capture.mark_ids():
            p = capture.mark_pose(m, side, kind)
            if p is not None:
                vals.append(float(p[2] * 1000.0))
        z_mm[side] = vals
    all_vals = [abs(v) for vals in z_mm.values() for v in vals]
    if not all_vals:
        return TableZReport(kind, z_mm, float("nan"), "FAIL")
    mx = max(all_vals)
    return TableZReport(kind, z_mm, mx, grade(mx, thresholds.table_z_max_mm, warn_factor=thresholds.warn_factor))


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_jitter(report: JitterReport) -> str:
    lines = [
        f"Static jitter ({report.kind}) — {report.duration_s:.1f} s @ {report.sample_rate_hz:.1f} Hz",
    ]
    for side, r in report.sides.items():
        sx, sy, sz = r.sigma_xyz_mm
        rr, rp, ry = r.sigma_rpy_deg
        lines.append(
            f"  {side:5s} [{r.status}] n={r.samples} tracked={r.tracked_fraction*100:5.1f}%  "
            f"σxyz=({sx:.2f},{sy:.2f},{sz:.2f}) mm  trans RMS={r.translation_rms_mm:.2f} mm "
            f"p95={r.translation_p95_mm:.2f} mm  σrpy=({rr:.3f},{rp:.3f},{ry:.3f})°  rot RMS={r.rotation_rms_deg:.3f}°"
        )
    lines.append(f"  => {report.status}")
    return "\n".join(lines)


def format_return(report: ReturnReport) -> str:
    lines = [f"Return-to-point ({report.side} {report.kind}) — reference mark {report.reference_mark}"]
    for m, e in zip(report.marks, report.errors_mm):
        lines.append(f"  mark {m:3d}: {e:6.2f} mm")
    if report.skipped_marks:
        lines.append(f"  skipped (untracked) marks: {report.skipped_marks}")
    lines.append(f"  mean={report.mean_mm:.2f} mm max={report.max_mm:.2f} mm => {report.status}")
    return "\n".join(lines)


def format_agreement(report: AgreementReport) -> str:
    lines = [f"Left/right agreement ({report.mode}, {report.kind})"]
    for p in report.pairs:
        if p.get("skipped"):
            lines.append(f"  {p.get('mark', p.get('marks'))}: skipped (untracked)")
        else:
            lines.append(f"  {p.get('mark', p.get('marks'))}: {p['separation_mm']:6.2f} mm")
    lines.append(f"  mean={report.mean_mm:.2f} mm max={report.max_mm:.2f} mm => {report.status}")
    return "\n".join(lines)


def format_table_z(report: TableZReport) -> str:
    lines = [f"Table z at marks ({report.kind})"]
    for side, vals in report.z_mm.items():
        lines.append(f"  {side:5s}: " + ", ".join(f"{v:+.1f}" for v in vals) + " mm")
    lines.append(f"  max |z|={report.max_abs_mm:.1f} mm => {report.status}")
    return "\n".join(lines)


def dump_json(report, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2))


__all__ = [
    "CSV_HEADER",
    "AgreementReport",
    "CaptureBuilder",
    "JitterReport",
    "ReturnReport",
    "TableZReport",
    "Thresholds",
    "TrackerCapture",
    "dump_json",
    "format_agreement",
    "format_jitter",
    "format_return",
    "format_table_z",
    "grade",
    "left_right_agreement",
    "return_to_point",
    "robust_mean_pose",
    "static_jitter",
    "table_z",
]
