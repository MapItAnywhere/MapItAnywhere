# Copyright (c) Meta Platforms, Inc. and affiliates.

# Adapted from OrienterNet, Paul-Edouard Sarlin, ETH Zurich
# https://github.com/facebookresearch/OrienterNet
# Released under the CC-BY-NC license

import numpy as np
import torch


def chunk_sequence(
    data,
    indices,
    *,
    names=None,
    max_length=100,
    min_length=1,
    max_delay_s=None,
    max_inter_dist=None,
    max_total_dist=None,
):
    sort_array = data.get("capture_time", data.get("index"))
    if sort_array is None:
        sort_array = indices if names is None else names
    indices = sorted(indices, key=lambda i: sort_array[i].tolist())
    centers = torch.stack([data["t_c2w"][i][:2] for i in indices]).numpy()
    dists = np.linalg.norm(np.diff(centers, axis=0), axis=-1)
    if "capture_time" in data:
        times = torch.stack([data["capture_time"][i] for i in indices])
        times = times.double() / 1e3  # ms to s
        delays = np.diff(times, axis=0)
    else:
        delays = np.zeros_like(dists)
    chunks = [[indices[0]]]
    dist_total = 0
    for dist, delay, idx in zip(dists, delays, indices[1:]):
        dist_total += dist
        if (
            (max_inter_dist is not None and dist > max_inter_dist)
            or (max_total_dist is not None and dist_total > max_total_dist)
            or (max_delay_s is not None and delay > max_delay_s)
            or len(chunks[-1]) >= max_length
        ):
            chunks.append([])
            dist_total = 0
        chunks[-1].append(idx)
    chunks = list(filter(lambda c: len(c) >= min_length, chunks))
    chunks = sorted(chunks, key=len, reverse=True)
    return chunks