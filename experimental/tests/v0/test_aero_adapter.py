"""Aero Hand Open adapter tests.

The load-bearing check: our tendon model must reproduce the actuation limit
table published in the official SDK (AeroHandConstants), because those limits
were derived by TetherIA by pushing the joint box through the same coupling.
"""

import numpy as np
import pytest

from ego_collector.hands3d.aero import (
    ACTUATION_LOWER_DEG,
    ACTUATION_UPPER_DEG,
    COMPACT_UPPER_DEG,
    AeroHandFK,
    canonical_to_aero,
    compact_to_full16_deg,
    default_urdf,
    full16_to_actuations_deg,
    grasp_to_canonical,
    human_semantic7,
    semantic7_normalized,
)


def _act_unclipped(joints16_deg):
    # bypass the clip by calling on a box corner and comparing pre-clip math
    from ego_collector.hands3d import aero as A

    q = np.radians(np.asarray(joints16_deg, dtype=np.float64))
    out = np.zeros(7)
    out[0] = np.degrees(q[0])
    out[1] = np.degrees((A.THUMB_FLEX_CMC_ABD * q[0] + A.THUMB_FLEX_CMC_FLEX * q[1]) / A.MOTOR_PULLEY_RADIUS)
    out[2] = np.degrees(
        (A.THUMB_IP_CMC_ABD * q[0] - A.THUMB_IP_CMC_FLEX * q[1] + A.THUMB_IP_MCP * q[2] + A.THUMB_IP_IP * q[3])
        / A.MOTOR_PULLEY_RADIUS
    )
    c1, c2, c3 = A.FINGER_COEFFS
    for k in range(4):
        f = q[4 + 3 * k : 7 + 3 * k]
        out[3 + k] = np.degrees((c1 * f[0] + c2 * f[1] + c3 * f[2]) / A.MOTOR_PULLEY_RADIUS)
    return out


class TestTendonModelMatchesSDK:
    def test_upper_limits(self):
        # SDK upper limits are the max of the actuation over the joint box:
        # thumb_tendon is maximised at cmc_flex = 0 (negative coefficient).
        corner = compact_to_full16_deg(np.array([100.0, 0.0, 90.0, 90.0, 90.0, 90.0, 90.0]))
        act = _act_unclipped(corner)
        assert act[0] == pytest.approx(100.0)
        assert act[2] == pytest.approx(ACTUATION_UPPER_DEG[2], abs=0.05)
        # cmc_flex act is maximised at abd = flex = max
        corner2 = compact_to_full16_deg(np.array(COMPACT_UPPER_DEG))
        act2 = _act_unclipped(corner2)
        assert act2[1] == pytest.approx(ACTUATION_UPPER_DEG[1], abs=0.05)
        for k in range(3, 7):
            assert act2[k] == pytest.approx(ACTUATION_UPPER_DEG[k], abs=0.05)

    def test_lower_limit_thumb_tendon(self):
        # min at abd=0, flex=max, curl=0
        corner = compact_to_full16_deg(np.array([0.0, 55.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        act = _act_unclipped(corner)
        assert act[2] == pytest.approx(ACTUATION_LOWER_DEG[2], abs=0.05)

    def test_output_is_clipped_to_hw_limits(self):
        _, _, act = canonical_to_aero(np.ones(7))
        assert np.all(act <= np.array(ACTUATION_UPPER_DEG) + 1e-9)
        _, _, act0 = canonical_to_aero(np.zeros(7))
        assert np.all(act0 >= np.array(ACTUATION_LOWER_DEG) - 1e-9)


class TestHumanSemantic7:
    @staticmethod
    def _flat_hand():
        """Synthetic open hand: fingers straight along +y, thumb 45deg off."""
        J = np.zeros((1, 21, 3))
        dirs = {
            "thumb": np.array([np.sin(np.radians(45)), np.cos(np.radians(45)), 0.0]),
            "index": np.array([0.15, 1.0, 0.0]),
            "middle": np.array([0.0, 1.0, 0.0]),
            "ring": np.array([-0.15, 1.0, 0.0]),
            "pinky": np.array([-0.3, 1.0, 0.0]),
        }
        from ego_collector.hands3d.aero import FINGER_IDX

        for f, ids in FINGER_IDX.items():
            d = dirs[f] / np.linalg.norm(dirs[f])
            for k, idx in enumerate(ids):
                J[0, idx] = d * 0.03 * (k + 1)
        return J

    def test_open_hand_maps_low(self):
        u = semantic7_normalized(human_semantic7(self._flat_hand()))
        # straight fingers -> all flexion dims ~0
        assert np.all(u[0, 2:] < 0.1)

    def test_curled_fingers_map_high(self):
        from ego_collector.hands3d.aero import FINGER_IDX

        J = self._flat_hand()
        # curl index: each successive bone rotates 75deg more around x-ish
        base = J[0, FINGER_IDX["index"][0]].copy()
        p = base.copy()
        d = np.array([0.0, 1.0, 0.0])
        pts = [p.copy()]
        for k in range(3):
            th = np.radians(75.0 * (k + 1))
            d_new = np.array([0.0, np.cos(th), -np.sin(th)])
            p = p + d_new * 0.03
            pts.append(p.copy())
        for idx, pt in zip(FINGER_IDX["index"][1:], pts[1:]):
            J[0, idx] = pt
        u = semantic7_normalized(human_semantic7(J))
        assert u[0, 3] > 0.8  # index_flex

    def test_shapes(self):
        J = np.zeros((5, 21, 3))
        J[:, :, 0] = np.linspace(0, 1, 21)[None, :]
        s = human_semantic7(J)
        assert s.shape == (5, 7)


class TestAeroV0Pinch:
    def test_grasp_scalar_monotone_pinch(self):
        u_open = grasp_to_canonical(np.array([0.0]))[0]
        u_closed = grasp_to_canonical(np.array([1.0]))[0]
        # thumb curl + index close, pinky stays relaxed
        assert u_closed[2] > u_open[2] + 0.5
        assert u_closed[3] > u_open[3] + 0.5
        assert abs(u_closed[6] - u_open[6]) < 0.2


class TestFK:
    def test_pinch_closes_thumb_index_gap(self):
        fk = AeroHandFK(default_urdf("right"), side="right")
        _, q_open, _ = canonical_to_aero(grasp_to_canonical(np.array([0.0]))[0])
        _, q_closed, _ = canonical_to_aero(grasp_to_canonical(np.array([1.0]))[0])
        d_open = np.linalg.norm(fk.fingertips(q_open)["thumb"] - fk.fingertips(q_open)["index"])
        d_closed = np.linalg.norm(fk.fingertips(q_closed)["thumb"] - fk.fingertips(q_closed)["index"])
        assert d_closed < d_open * 0.5
        assert d_closed < 0.035  # pinch gap under 3.5cm

    def test_all_links_resolved(self):
        fk = AeroHandFK(default_urdf("left"), side="left")
        pos = fk.link_positions(np.zeros(16))
        for chain in fk.chains():
            for link in chain:
                assert link in pos, link


class TestThumbGapServo:
    def test_servo_matches_target_aperture(self):
        from ego_collector.hands3d.aero import thumb_gap_servo

        fk = AeroHandFK(default_urdf("right"), side="right")
        u = np.tile(np.array([0.5, 0.4, 0.5, 0.3, 0.3, 0.3, 0.3]), (3, 1))
        # at this pose the curl-only reachable gap envelope is ~61-121mm;
        # targets outside it clamp to the nearest achievable gap
        targets = np.array([0.065, 0.09, 0.11])
        out = thumb_gap_servo(u, targets, fk)
        for row, tgt in zip(out, targets):
            _, q16, _ = canonical_to_aero(row)
            tips = fk.fingertips(q16)
            gap = np.linalg.norm(tips["thumb"] - tips["index"])
            assert abs(gap - tgt) < 0.006  # within 6mm inside the envelope

    def test_unreachable_target_clamps_to_envelope_edge(self):
        from ego_collector.hands3d.aero import thumb_gap_servo

        fk = AeroHandFK(default_urdf("right"), side="right")
        u = np.array([[0.5, 0.4, 0.5, 0.3, 0.3, 0.3, 0.3]])
        out = thumb_gap_servo(u.copy(), np.array([0.0]), fk)  # 0mm: unreachable
        _, q16, _ = canonical_to_aero(out[0])
        tips = fk.fingertips(q16)
        gap = np.linalg.norm(tips["thumb"] - tips["index"])
        assert gap < 0.07  # sits at the closed edge of the envelope (~61mm)

    def test_only_thumb_curl_changes(self):
        from ego_collector.hands3d.aero import thumb_gap_servo

        fk = AeroHandFK(default_urdf("left"), side="left")
        u = np.tile(np.array([0.5, 0.4, 0.5, 0.3, 0.3, 0.3, 0.3]), (2, 1))
        out = thumb_gap_servo(u, np.array([0.05, 0.08]), fk)
        keep = [0, 1, 3, 4, 5, 6]
        assert np.allclose(out[:, keep], u[:, keep])

    def test_nan_aperture_keeps_semantic_value(self):
        from ego_collector.hands3d.aero import thumb_gap_servo

        fk = AeroHandFK(default_urdf("right"), side="right")
        u = np.tile(np.array([0.5, 0.4, 0.5, 0.3, 0.3, 0.3, 0.3]), (2, 1))
        out = thumb_gap_servo(u, np.array([np.nan, 0.06]), fk)
        assert out[0, 2] == u[0, 2]
