"""Rotation validity of stored rot6d (spec 31): reconstruct R by Gram-Schmidt and also check the RAW stored rows are
already orthonormal (a stored rot6d that only becomes valid after Gram-Schmidt is a bug upstream)."""
import numpy as np
from ..geometry.rotation6d import rotation_6d_to_matrix


def rotation_report(d6, tol=1e-4):
    d6 = np.asarray(d6, np.float64); d6 = d6[np.isfinite(d6).all(1)]
    R = rotation_6d_to_matrix(d6)
    orth = np.abs(np.swapaxes(R, -1, -2) @ R - np.eye(3)).max(axis=(1, 2)); det = np.abs(np.linalg.det(R) - 1)
    a, b = d6[:, :3], d6[:, 3:]
    raw = np.maximum.reduce([np.abs(np.linalg.norm(a, axis=1) - 1), np.abs(np.linalg.norm(b, axis=1) - 1), np.abs(np.sum(a * b, 1))])
    bad = (orth > tol) | (det > tol) | (raw > tol)
    return dict(n=int(len(d6)), invalid_rotation_count=int(bad.sum()), max_orthogonality_error=float(orth.max()),
                max_determinant_error=float(det.max()), max_raw_rot6d_error=float(raw.max()))
