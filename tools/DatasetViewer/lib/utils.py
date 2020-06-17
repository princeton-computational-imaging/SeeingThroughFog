import numpy as np
from pyquaternion import Quaternion


def quaternion2rotmatrix(quaternion):
    """Calculate rotation matrix R from quaternion values - brief introduction into quaternions can be found at
    http://www.cs.ucr.edu/~vbz/resources/quatut.pdf (cf. pg. 6, 7)"""

    w, x, y, z = quaternion['qw'], quaternion['qx'], quaternion['qy'], quaternion['qz']
    R = Quaternion(w=w * 360 / 2 / np.pi, x=x * 360 / 2 / np.pi,
                   y=y * 360 / 2 / np.pi, z=z * 360 / 2 / np.pi)
    return R.rotation_matrix


def rotx_matrix(roty):
    c = np.cos(roty)
    s = np.sin(roty)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty_matrix(roty):
    c = np.cos(roty)
    s = np.sin(roty)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotz_matrix(roty):
    c = np.cos(roty)
    s = np.sin(roty)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
