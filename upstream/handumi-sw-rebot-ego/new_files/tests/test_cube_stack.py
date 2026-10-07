from __future__ import annotations

import itertools

from handumi.tasks.cube_stack import (
    DEFAULT_FIXED_SLOTS,
    DEFAULT_PAN_XY,
    MIN_CUBE_SPACING_M,
    all_orders,
    instruction_for,
    make_plans,
    read_plans,
    write_plans,
)


def test_orders_and_instruction():
    assert len(all_orders()) == 6
    assert instruction_for(["red", "blue", "green"]) == "Stack red, blue, green."
    assert instruction_for(["green", "blue", "red"], 2) == "Put green at the bottom, blue in the middle, and red on top."


def test_regime_a_is_fixed_and_orders_balanced():
    plans = make_plans(regime="A", episodes=12, seed=3)
    assert all(p.cube_xy == {k: tuple(v) for k, v in DEFAULT_FIXED_SLOTS.items()} for p in plans)
    assert all(p.pan_xy == DEFAULT_PAN_XY for p in plans)
    counts = {}
    for p in plans:
        counts["-".join(p.order)] = counts.get("-".join(p.order), 0) + 1
    assert set(counts.values()) == {2}  # 12 episodes / 6 orders


def test_regime_b_c_respect_spacing_and_are_seeded(tmp_path):
    for regime in ("B", "C"):
        plans = make_plans(regime=regime, episodes=30, seed=7)
        for p in plans:
            pts = list(p.cube_xy.values())
            for a, b in itertools.combinations(pts, 2):
                assert ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 >= MIN_CUBE_SPACING_M - 1e-9
            if regime == "B":
                assert p.pan_xy == DEFAULT_PAN_XY
        again = make_plans(regime=regime, episodes=30, seed=7)
        assert [p.to_dict() for p in plans] == [p.to_dict() for p in again]
    path = tmp_path / "plans.json"
    write_plans(plans, path)
    back = read_plans(path)
    assert [p.to_dict() for p in back] == [p.to_dict() for p in plans]
