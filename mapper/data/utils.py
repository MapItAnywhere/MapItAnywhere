# Copyright (c) Meta Platforms, Inc. and affiliates.

# Adapted from OrienterNet, Paul-Edouard Sarlin, ETH Zurich
# https://github.com/facebookresearch/OrienterNet
# Released under the CC-BY-NC license

import numpy as np
from scipy.spatial.transform import Rotation


def crop_map(raster, xy, size, seed=None):
    h, w = raster.shape[-2:]
    state = np.random.RandomState(seed)
    top = state.randint(0, h - size + 1)
    left = state.randint(0, w - size + 1)
    raster = raster[..., top : top + size, left : left + size]
    xy -= np.array([left, top])
    return raster, xy


def decompose_rotmat(R_c2w):
    R_cv2xyz = Rotation.from_euler("X", -90, degrees=True)
    rot_w2c = R_cv2xyz * Rotation.from_matrix(R_c2w).inv()
    roll, pitch, yaw = rot_w2c.as_euler("YXZ", degrees=True)
    return roll, pitch, yaw