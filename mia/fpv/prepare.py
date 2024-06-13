# Adapted from prepare.py

import asyncio
import argparse
from collections import defaultdict
import json
import shutil
from pathlib import Path
from typing import List, Dict

import numpy as np
import cv2
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from omegaconf import DictConfig, OmegaConf
from opensfm.pygeometry import Camera
from opensfm.pymap import Shot
from opensfm.undistort import (
    perspective_camera_from_fisheye,
    perspective_camera_from_perspective,
)

from .. import logger
# from ...osm.tiling import TileManager
# from ...osm.viz import GeoPlotter
from .geo import BoundaryBox, Projection
from .utils import decompose_rotmat
from .utils_sfm import (
    keyframe_selection,
    perspective_camera_from_pano,
    scale_camera,
    CameraUndistorter,
    PanoramaUndistorter,
    undistort_shot,
)
from .download import (
    opensfm_shot_from_info,
    image_filename,
)


default_cfg = OmegaConf.create(
    {
        "max_image_size": 512,
        "do_legacy_pano_offset": True,
        "min_dist_between_keyframes": 4,
        "tiling": {
            "tile_size": 128,
            "margin": 128,
            "ppm": 2,
        },
    }
)


def get_pano_offset(image_info: dict, do_legacy: bool = False) -> float:
    if do_legacy:
        seed = int(image_info["sfm_cluster"]["id"])
    else:
        seed = image_info["sequence"].__hash__()
    seed = seed % (2**32 - 1)
    return np.random.RandomState(seed).uniform(-45, 45)


def process_shot(
    shot: Shot, info: dict, image_path: Path, output_dir: Path, cfg: DictConfig
) -> List[Shot]:
    if not image_path.exists():
        logger.warn(f"Image {image_path} does not exist !")
        return None

    image_orig = cv2.imread(str(image_path))
    max_size = cfg.max_image_size
    pano_offset = None

    camera = shot.camera
    camera.width, camera.height = image_orig.shape[:2][::-1]
    if camera.is_panorama(camera.projection_type):
        camera_new = perspective_camera_from_pano(camera, max_size)
        undistorter = PanoramaUndistorter(camera, camera_new)
        pano_offset = get_pano_offset(info, cfg.do_legacy_pano_offset)
    elif camera.projection_type in ["fisheye", "perspective"]:
        if camera.projection_type == "fisheye":
            camera_new = perspective_camera_from_fisheye(camera)
        else:
            camera_new = perspective_camera_from_perspective(camera)
        camera_new = scale_camera(camera_new, max_size)
        camera_new.id = camera.id + "_undistorted"
        undistorter = CameraUndistorter(camera, camera_new)
    else:
        raise NotImplementedError(camera.projection_type)

    shots_undist, images_undist = undistort_shot(
        image_orig, shot, undistorter, pano_offset
    )
    for shot, image in zip(shots_undist, images_undist):
        cv2.imwrite(str(output_dir / f"{shot.id}.jpg"), image)

    return shots_undist


def pack_shot_dict(shot: Shot, info: dict) -> dict:
    latlong = info["computed_geometry.coordinates"][::-1]
    latlong_gps = info["geometry.coordinates"][::-1]
    w_p_c = shot.pose.get_origin()
    w_r_c = shot.pose.get_R_cam_to_world()
    rpy = decompose_rotmat(w_r_c)
    return dict(
        camera_id=shot.camera.id,
        latlong=latlong,
        t_c2w=w_p_c,
        R_c2w=w_r_c,
        roll_pitch_yaw=rpy,
        capture_time=info["captured_at"],
        gps_position=np.r_[latlong_gps, info["altitude"]],
        compass_angle=info["compass_angle"],
        chunk_id=int(info["sfm_cluster.id"]),
    )


def pack_camera_dict(camera: Camera) -> dict:
    assert camera.projection_type == "perspective"
    K = camera.get_K_in_pixel_coordinates(camera.width, camera.height)
    return dict(
        id=camera.id,
        model="PINHOLE",
        width=camera.width,
        height=camera.height,
        params=K[[0, 1, 0, 1], [0, 1, 2, 2]],
    )


def process_sequence(
    image_ids: List[int],
    image_infos: dict,
    projection: Projection,
    cfg: DictConfig,
    raw_image_dir: Path,
    out_image_dir: Path,
):
    shots = []
    dump = {}
    processed_ids = list()
    if len(image_ids) == 0:
        return dump, processed_ids

    image_ids = sorted(image_ids, key=lambda i: image_infos[i]["captured_at"])
    for i in image_ids:
        _, shot = opensfm_shot_from_info(image_infos[i], projection)
        shots.append(shot)
    shot_idxs = keyframe_selection(shots, min_dist=cfg.min_dist_between_keyframes)
    shots = [shots[i] for i in shot_idxs]

    shots_out = thread_map(
        lambda shot: process_shot(
            shot,
            image_infos[shot.id],
            raw_image_dir / image_filename.format(image_id=shot.id),
            out_image_dir,
            cfg,
        ),
        shots,
        disable=True,
    )
    shots_out = [s for s in shots_out if s is not None]
    shots_out = [(i, s) for i, ss in enumerate(shots_out) for s in ss if ss is not None]

    for index, shot in shots_out:
        i, suffix = shot.id.rsplit("_", 1)
        processed_ids.append(i)
        info = image_infos[i]
        seq_id = info["sequence"]
        is_pano = not suffix.endswith("undistorted")
        if is_pano:
            seq_id += f"_{suffix}"
        if seq_id not in dump:
            dump[seq_id] = dict(views={}, cameras={})

        view = pack_shot_dict(shot, info)
        view["index"] = index
        dump[seq_id]["views"][shot.id] = view
        dump[seq_id]["cameras"][shot.camera.id] = pack_camera_dict(shot.camera)
    return dump, processed_ids