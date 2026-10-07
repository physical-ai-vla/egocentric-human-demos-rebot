import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from ego_teleop.transforms.frames import FrameMapperConfig, HumanRobotFrameMapper, basis_matrix
from ego_teleop.transforms.se3 import make_T
from .conftest import T_of


def test_identity_mapper_is_identity():
    m = HumanRobotFrameMapper()
    D = T_of((0.1, 0.2, -0.3), (10, -20, 30))
    assert np.allclose(m.map_delta(D), D)


def test_permutation_and_sign_is_orthogonal_and_maps_translation():
    cfg = FrameMapperConfig(translation_axis_map=("y", "z", "x"), translation_sign=(-1, 1, 1))
    m = HumanRobotFrameMapper(cfg)
    assert np.allclose(m.M_t @ m.M_t.T, np.eye(3))
    assert np.allclose(m.map_translation([1, 2, 3]), [-2, 3, 1])       # robot x <- -human y, robot y <- human z, robot z <- human x


def test_rotation_conjugation_preserves_composition_and_properness():
    cfg = FrameMapperConfig(translation_axis_map=("x", "z", "y"), translation_sign=(1, 1, 1))   # one axis swap: det = -1 (mirror, left hand -> right arm)
    m = HumanRobotFrameMapper(cfg)
    assert m.is_mirror
    A, B = T_of((0.1, 0, 0), (0, 30, 0)), T_of((0, 0.05, 0), (15, 0, 40))
    assert np.allclose(m.map_delta(A @ B), m.map_delta(A) @ m.map_delta(B), atol=1e-9)
    R = m.map_rotation(A[:3, :3]); assert np.isclose(np.linalg.det(R), 1.0) and np.allclose(R @ R.T, np.eye(3))


def test_scale_applies_to_translation_only_and_rotation_scale_to_angle():
    m = HumanRobotFrameMapper(FrameMapperConfig(translation_scale=(0.5, 0.5, 2.0), rotation_scale=0.5))
    D = m.map_delta(T_of((0.1, 0.1, 0.1), (0, 0, 40)))
    assert np.allclose(D[:3, 3], [0.05, 0.05, 0.2])
    assert np.isclose(np.degrees(Rotation.from_matrix(D[:3, :3]).magnitude()), 20.0)


@pytest.mark.parametrize("amap,sign", [(("x", "x", "z"), (1, 1, 1)), (("x", "y"), (1, 1)), (("x", "y", "z"), (1, 0.5, 1))])
def test_invalid_maps_rejected(amap, sign):
    with pytest.raises(ValueError):
        basis_matrix(amap, sign)


def test_yaml_roundtrip(tmp_path):
    import yaml
    cfg = FrameMapperConfig(translation_axis_map=("z", "x", "y"), translation_sign=(1, -1, 1), translation_scale=(0.8, 0.8, 0.8))
    p = tmp_path / "frames.yaml"; p.write_text(yaml.safe_dump(cfg.to_dict()))
    assert FrameMapperConfig.from_yaml(p) == cfg
