import numpy as np
from scipy.spatial.transform import Rotation as R


def quat_to_rot(q):  # [w, x, y, z]
    from scipy.spatial.transform import Rotation as R

    return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def delta_q(theta: np.ndarray) -> R:
    """
    Computes the small angle quaternion (delta quaternion).
    Assumes theta is a small rotation vector.
    """
    dq = np.zeros(4)
    dq[0] = 1.0
    dq[1:] = 0.5 * theta
    return R.from_quat([dq[1], dq[2], dq[3], dq[0]])  # [x, y, z, w]


def qleft(q: R) -> np.ndarray:
    """
    Returns the left quaternion multiplication matrix for q.
    Used for: q ⊗ p = Qleft(q) @ p_quat
    """
    q = q.as_quat()  # [x, y, z, w]
    x, y, z, w = q
    return np.array([[w, -z, y, x], [z, w, -x, y], [-y, x, w, z], [-x, -y, -z, w]])


def qright(q: R) -> np.ndarray:
    """
    Returns the right quaternion multiplication matrix for q.
    Used for: q ⊗ p = Qright(p) @ q_quat
    """
    q = q.as_quat()  # [x, y, z, w]
    x, y, z, w = q
    return np.array([[w, z, -y, x], [-z, w, x, y], [y, -x, w, z], [-x, -y, -z, w]])


def positify(quat: np.ndarray) -> np.ndarray:
    """
    Ensure quaternion has a non-negative scalar part.
    Input: quat as np.array([w, x, y, z])
    """
    return quat if quat[0] >= 0 else -quat


def local_size(size: int) -> int:
    return 6 if size == 7 else size


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    Create a skew-symmetric matrix from a 3-element vector.
    :param v: A 3D vector (numpy array of shape (3,))
    :return: A 3x3 skew-symmetric matrix
    """
    assert v.shape == (3,)
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def g_to_r(acc: np.ndarray) -> np.ndarray:
    """
    Align the input gravity vector to Z+, using quaternion FromTwoVectors logic.
    Then apply a yaw compensation to make yaw = 0.
    Equivalent to Utility::g2R().
    """
    acc_norm = acc / np.linalg.norm(acc)
    target = np.array([0.0, 0.0, 1.0])

    # Equivalent of Eigen::Quaterniond::FromTwoVectors()
    v = np.cross(acc_norm, target)
    c = np.dot(acc_norm, target)
    if c < -0.999999:
        # Opposite vectors
        R_align = -np.eye(3)
    elif c > 0.999999:
        R_align = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R_align = np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v) ** 2))

    yaw = r2ypr(R_align)[0]  # in degrees
    R_out = ypr2r(np.array([-yaw, 0, 0])) @ R_align
    return R_out


def r2ypr(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to yaw-pitch-roll (in degrees).
    Equivalent to Utility::R2ypr() in VINS-Fusion.
    """
    n = R[:, 0]
    o = R[:, 1]
    a = R[:, 2]

    yaw = np.arctan2(n[1], n[0])
    pitch = np.arctan2(-n[2], n[0] * np.cos(yaw) + n[1] * np.sin(yaw))
    roll = np.arctan2(
        a[0] * np.sin(yaw) - a[1] * np.cos(yaw),
        -o[0] * np.sin(yaw) + o[1] * np.cos(yaw),
    )

    return np.array([yaw, pitch, roll]) * 180.0 / np.pi


def ypr2r(ypr: np.ndarray) -> np.ndarray:
    """
    Convert yaw-pitch-roll (in degrees) to a 3x3 rotation matrix.
    Equivalent to Utility::ypr2R().
    """
    y, p, r = np.radians(ypr)

    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])

    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])

    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])

    return Rz @ Ry @ Rx
