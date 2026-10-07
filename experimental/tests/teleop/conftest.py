import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from ego_teleop.transforms.se3 import make_T


def T_of(xyz=(0, 0, 0), rotvec_deg=(0, 0, 0)):
    return make_T(Rotation.from_rotvec(np.radians(rotvec_deg)).as_matrix(), xyz)


@pytest.fixture
def T_of_fixture():
    return T_of
