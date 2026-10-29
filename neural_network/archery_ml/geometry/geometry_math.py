"""Checked projection shared by fitting, synthetic truth, and diagnostics."""
import numpy as np


def project(points, matrix):
    homogeneous = np.column_stack((np.asarray(points), np.ones(len(points)))) @ matrix.T
    if not np.isfinite(homogeneous).all() or np.any(np.abs(homogeneous[:, 2]) < 1e-10):
        raise np.linalg.LinAlgError("points at projective horizon")
    return homogeneous[:, :2] / homogeneous[:, 2:]
