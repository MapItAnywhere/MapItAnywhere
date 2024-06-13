# Copyright (c) Meta Platforms, Inc. and affiliates.

# Adapted from Hierarchical-Localization, Paul-Edouard Sarlin, ETH Zurich
# https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/utils/viz.py
# Released under the Apache License 2.0

import numpy as np
import torch 


def features_to_RGB(*Fs, masks=None, skip=1):
    """Project a list of d-dimensional feature maps to RGB colors using PCA."""
    from sklearn.decomposition import PCA

    def normalize(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    if masks is not None:
        assert len(Fs) == len(masks)

    flatten = []
    for i, F in enumerate(Fs):
        c, h, w = F.shape
        F = np.rollaxis(F, 0, 3)
        F_flat = F.reshape(-1, c)
        if masks is not None and masks[i] is not None:
            mask = masks[i]
            assert mask.shape == F.shape[:2]
            F_flat = F_flat[mask.reshape(-1)]
        flatten.append(F_flat)
    flatten = np.concatenate(flatten, axis=0)
    flatten = normalize(flatten)

    pca = PCA(n_components=3)
    if skip > 1:
        pca.fit(flatten[::skip])
        flatten = pca.transform(flatten)
    else:
        flatten = pca.fit_transform(flatten)
    flatten = (normalize(flatten) + 1) / 2

    Fs_rgb = []
    for i, F in enumerate(Fs):
        h, w = F.shape[-2:]
        if masks is None or masks[i] is None:
            F_rgb, flatten = np.split(flatten, [h * w], axis=0)
            F_rgb = F_rgb.reshape((h, w, 3))
        else:
            F_rgb = np.zeros((h, w, 3))
            indices = np.where(masks[i])
            F_rgb[indices], flatten = np.split(flatten, [len(indices[0])], axis=0)
            F_rgb = np.concatenate([F_rgb, masks[i][..., None]], axis=-1)
        Fs_rgb.append(F_rgb)
    assert flatten.shape[0] == 0, flatten.shape
    return Fs_rgb


def one_hot_argmax_to_rgb(y, num_class):
    '''
    Args:
        probs: (B, C, H, W)
        num_class: int
        0: road 0
1: crossing 1
2: explicit_pedestrian 2
4: building 
6: terrain
7: parking `

    '''


    class_colors = {
        'road': (0, 0, 0),           # 0: Black
        'crossing': (255, 0, 0),     # 1; Red
        'explicit_pedestrian': (255, 255, 0),  # 2: Yellow
        # 'explicit_void': (128, 128, 128),      # 3: White
        'park': (0, 255, 0),         # 4: Green
        'building': (255, 0, 255),   # 5: Magenta
        'water': (0, 0, 255),        # 6: Blue
        'terrain': (0, 255, 255),    # 7: Cyan
        'parking': (170, 170, 170),  # 8: Dark Grey
        'train': (85, 85, 85) ,       # 9: Light Grey
        'predicted_void': (256, 256, 256)
    }
    class_colors = class_colors.values()
    class_colors = [torch.tensor(x) for x in class_colors]

    argmaxed = torch.argmax((y > 0.5).float(), dim=1) # Take argmax
    argmaxed[torch.all(y <= 0.5, dim=1)] = num_class
    # print(argmaxed.shape)

    seg_rgb = torch.ones(
        (
            argmaxed.shape[0],
            3,
            argmaxed.shape[1],
            argmaxed.shape[2],
        )
    ) * 256
    for i in range(num_class + 1):
        seg_rgb[:, 0, :, :][argmaxed == i] = class_colors[i][0]
        seg_rgb[:, 1, :, :][argmaxed == i] = class_colors[i][1]
        seg_rgb[:, 2, :, :][argmaxed == i] = class_colors[i][2]

    return seg_rgb
