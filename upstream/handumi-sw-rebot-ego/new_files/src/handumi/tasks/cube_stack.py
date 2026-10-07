"""3-cube stacking episode plans (Phase 9 of the reBot ego-bimanual pipeline).

Every recorded episode needs (a) a stack order, (b) where the cubes and the pan
start, and (c) the language instruction that goes into ``--task``. This module
makes those decisions reproducibly (seeded) and logs them as JSON so the
dataset-A/B/C splits can be audited later.

Dataset regimes::

    A  cube positions fixed,      pan fixed,      order randomized
    B  cube positions randomized, pan fixed,      order randomized
    C  cube positions randomized, pan randomized, order randomized

Coordinates are in the HandUMI table frame (metres; +X right, +Y away from the
operator, origin = ChArUco board centre).
"""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

Regime = Literal["A", "B", "C"]
DEFAULT_COLORS: tuple[str, ...] = ("red", "blue", "green")

# Fixed layout used by regime A (and as the pan default for A/B). Slots are
# spread 15 cm apart so grasp-FK based cube localisation stays unambiguous
# (cube positions recovered from teleop grasps carry ~5 cm noise).
DEFAULT_FIXED_SLOTS: dict[str, tuple[float, float]] = {
    "red": (-0.15, 0.20),
    "blue": (0.00, 0.20),
    "green": (0.15, 0.20),
}
DEFAULT_PAN_XY: tuple[float, float] = (0.0, 0.38)
# Random placement region for cubes (regime B/C) and the pan (regime C).
DEFAULT_CUBE_REGION = ((-0.25, 0.25), (0.10, 0.30))  # (x_min,x_max),(y_min,y_max)
DEFAULT_PAN_REGION = ((-0.15, 0.15), (0.32, 0.45))
MIN_CUBE_SPACING_M = 0.12
MIN_PAN_CLEARANCE_M = 0.14

INSTRUCTION_TEMPLATES: tuple[str, ...] = (
    "Stack {a}, {b}, {c}.",
    "Stack the {a} cube, then the {b} cube, then the {c} cube in the pan.",
    "Put {a} at the bottom, {b} in the middle, and {c} on top.",
    "Place the {a} cube in the pan, then stack {b} on it and {c} on top.",
)


@dataclass
class CubePlan:
    episode: int
    regime: Regime
    order: list[str]  # bottom -> top
    cube_xy: dict[str, tuple[float, float]]
    pan_xy: tuple[float, float]
    instruction: str
    template_index: int
    seed: int
    colors: list[str] = field(default_factory=lambda: list(DEFAULT_COLORS))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cube_xy"] = {k: list(v) for k, v in self.cube_xy.items()}
        d["pan_xy"] = list(self.pan_xy)
        return d

    def layout_text(self) -> str:
        lines = [f"episode {self.episode:03d}  regime {self.regime}  order {' -> '.join(self.order)} (bottom->top)"]
        for c in self.colors:
            x, y = self.cube_xy[c]
            lines.append(f"  {c:6s} cube at x={x:+.2f} y={y:+.2f}")
        lines.append(f"  pan at x={self.pan_xy[0]:+.2f} y={self.pan_xy[1]:+.2f}")
        lines.append(f'  task: "{self.instruction}"')
        return "\n".join(lines)


def instruction_for(order: list[str], template_index: int = 0) -> str:
    a, b, c = order
    return INSTRUCTION_TEMPLATES[template_index % len(INSTRUCTION_TEMPLATES)].format(a=a, b=b, c=c)


def all_orders(colors: tuple[str, ...] = DEFAULT_COLORS) -> list[list[str]]:
    return [list(p) for p in itertools.permutations(colors)]


def _sample_xy(rng: random.Random, region) -> tuple[float, float]:
    (x0, x1), (y0, y1) = region
    return (round(rng.uniform(x0, x1), 3), round(rng.uniform(y0, y1), 3))


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def sample_layout(
    rng: random.Random,
    *,
    regime: Regime,
    colors: tuple[str, ...] = DEFAULT_COLORS,
    fixed_slots: dict[str, tuple[float, float]] = DEFAULT_FIXED_SLOTS,
    pan_default: tuple[float, float] = DEFAULT_PAN_XY,
    cube_region=DEFAULT_CUBE_REGION,
    pan_region=DEFAULT_PAN_REGION,
    min_spacing: float = MIN_CUBE_SPACING_M,
    min_pan_clearance: float = MIN_PAN_CLEARANCE_M,
    max_tries: int = 500,
) -> tuple[dict[str, tuple[float, float]], tuple[float, float]]:
    pan = _sample_xy(rng, pan_region) if regime == "C" else tuple(pan_default)
    if regime == "A":
        return {c: tuple(fixed_slots[c]) for c in colors}, pan
    for _ in range(max_tries):
        pts = [_sample_xy(rng, cube_region) for _ in colors]
        ok = all(_dist(p, q) >= min_spacing for p, q in itertools.combinations(pts, 2))
        ok = ok and all(_dist(p, pan) >= min_pan_clearance for p in pts)
        if ok:
            return dict(zip(colors, pts)), pan
    raise RuntimeError("could not sample a non-overlapping cube layout; loosen the region/spacing")


def make_plans(
    *,
    regime: Regime,
    episodes: int,
    seed: int = 0,
    start_episode: int = 0,
    colors: tuple[str, ...] = DEFAULT_COLORS,
    balanced_orders: bool = True,
    template_index: int | None = 0,
) -> list[CubePlan]:
    """Generate ``episodes`` plans; orders are balanced over the 6 permutations."""
    rng = random.Random(seed)
    orders = all_orders(colors)
    plans: list[CubePlan] = []
    deck: list[list[str]] = []
    for i in range(episodes):
        if balanced_orders:
            if not deck:
                deck = [list(o) for o in orders]
                rng.shuffle(deck)
            order = deck.pop()
        else:
            order = list(rng.choice(orders))
        cube_xy, pan_xy = sample_layout(rng, regime=regime, colors=colors)
        t_idx = rng.randrange(len(INSTRUCTION_TEMPLATES)) if template_index is None else template_index
        plans.append(
            CubePlan(
                episode=start_episode + i,
                regime=regime,
                order=order,
                cube_xy=cube_xy,
                pan_xy=pan_xy,
                instruction=instruction_for(order, t_idx),
                template_index=t_idx,
                seed=seed,
                colors=list(colors),
            )
        )
    return plans


def write_plans(plans: list[CubePlan], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([p.to_dict() for p in plans], indent=2))


def read_plans(path: Path) -> list[CubePlan]:
    data = json.loads(Path(path).read_text())
    out = []
    for d in data:
        d = dict(d)
        d["cube_xy"] = {k: tuple(v) for k, v in d["cube_xy"].items()}
        d["pan_xy"] = tuple(d["pan_xy"])
        out.append(CubePlan(**d))
    return out


__all__ = [
    "CubePlan",
    "DEFAULT_COLORS",
    "INSTRUCTION_TEMPLATES",
    "all_orders",
    "instruction_for",
    "make_plans",
    "read_plans",
    "sample_layout",
    "write_plans",
]
