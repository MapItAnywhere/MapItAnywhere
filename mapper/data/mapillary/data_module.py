import json
from collections import defaultdict
import os
import shutil
import tarfile
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as torchdata
from omegaconf import DictConfig

from ... import logger
from .dataset import MapLocDataset
from ..sequential import chunk_sequence
from ..torch import collate, worker_init_fn
from ..schema import MIADataConfiguration

def pack_dump_dict(dump):
    for per_seq in dump.values():
        if "points" in per_seq:
            for chunk in list(per_seq["points"]):
                points = per_seq["points"].pop(chunk)
                if points is not None:
                    per_seq["points"][chunk] = np.array(
                        per_seq["points"][chunk], np.float64
                    )
        for view in per_seq["views"].values():
            for k in ["R_c2w", "roll_pitch_yaw"]:
                view[k] = np.array(view[k], np.float32)
            for k in ["chunk_id"]:
                if k in view:
                    view.pop(k)
        if "observations" in view:
            view["observations"] = np.array(view["observations"])
        for camera in per_seq["cameras"].values():
            for k in ["params"]:
                camera[k] = np.array(camera[k], np.float32)
    return dump


class MapillaryDataModule(pl.LightningDataModule):
    dump_filename = "dump.json"
    images_archive = "images.tar.gz"
    images_dirname = "images/"
    semantic_masks_dirname = "semantic_masks/"
    flood_dirname = "flood_fill/"

    def __init__(self, cfg: MIADataConfiguration):
        super().__init__()
        self.cfg = cfg
        self.root = self.cfg.data_dir
        self.local_dir = None

    def prepare_data(self):
        for scene in self.cfg.scenes:
            dump_dir = self.root / scene
            assert (dump_dir / self.dump_filename).exists(), dump_dir
            # assert (dump_dir / self.cfg.tiles_filename).exists(), dump_dir
            if self.local_dir is None:
                assert (dump_dir / self.images_dirname).exists(), dump_dir
                continue
            assert (dump_dir / self.semantic_masks_dirname).exists(), dump_dir
            assert (dump_dir / self.flood_dirname).exists(), dump_dir
            # Cache the folder of images locally to speed up reading
            local_dir = self.local_dir / scene
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(exist_ok=True, parents=True)
            images_archive = dump_dir / self.images_archive
            logger.info("Extracting the image archive %s.", images_archive)
            with tarfile.open(images_archive) as fp:
                fp.extractall(local_dir)

    def setup(self, stage: Optional[str] = None):
        self.dumps = {}
        # self.tile_managers = {}
        self.image_dirs = {}
        self.seg_masks_dir = {}
        self.flood_masks_dir = {}
        names = []

        for scene in self.cfg.scenes:
            logger.info("Loading scene %s.", scene)
            dump_dir = self.root / scene

            logger.info("Loading dump json file %s.", self.dump_filename)
            with (dump_dir / self.dump_filename).open("r") as fp:
                self.dumps[scene] = pack_dump_dict(json.load(fp))
            for seq, per_seq in self.dumps[scene].items():
                for cam_id, cam_dict in per_seq["cameras"].items():
                    if cam_dict["model"] != "PINHOLE":
                        raise ValueError(
                            f"Unsupported camera model: {cam_dict['model']} for {scene},{seq},{cam_id}"
                        )

            self.image_dirs[scene] = (
                (self.local_dir or self.root) / scene / self.images_dirname
            )
            assert self.image_dirs[scene].exists(), self.image_dirs[scene]

            self.seg_masks_dir[scene] = (
                (self.local_dir or self.root) / scene / self.semantic_masks_dirname
            )   
            assert self.seg_masks_dir[scene].exists(), self.seg_masks_dir[scene]

            self.flood_masks_dir[scene] = (
                (self.local_dir or self.root) / scene / self.flood_dirname
            )   
            assert self.flood_masks_dir[scene].exists(), self.flood_masks_dir[scene]

            images = set(x.split('.')[0] for x in os.listdir(self.image_dirs[scene]))
            flood_masks = set(x.split('.')[0] for x in os.listdir(self.flood_masks_dir[scene]))
            semantic_masks = set(x.split('.')[0] for x in os.listdir(self.seg_masks_dir[scene]))

            for seq, data in self.dumps[scene].items():
                for name in data["views"]:
                    if name in images and name.split("_")[0] in flood_masks and name.split("_")[0] in semantic_masks:
                        names.append((scene, seq, name))
 
        self.parse_splits(self.cfg.split, names)
        if self.cfg.filter_for is not None:
            self.filter_elements()
        self.pack_data()

    def pack_data(self):
        # We pack the data into compact tensors that can be shared across processes without copy
        exclude = {
            "compass_angle",
            "compass_accuracy",
            "gps_accuracy",
            "chunk_key",
            "panorama_offset",
        }
        cameras = {
            scene: {seq: per_seq["cameras"] for seq, per_seq in per_scene.items()}
            for scene, per_scene in self.dumps.items()
        }
        points = {
            scene: {
                seq: {
                    i: torch.from_numpy(p) for i, p in per_seq.get("points", {}).items()
                }
                for seq, per_seq in per_scene.items()
            }
            for scene, per_scene in self.dumps.items()
        }
        self.data = {}

        # TODO: remove
        if self.cfg.split == "splits_MGL_13loc.json":
        # Use Last 20% as Val
            num_samples_to_move = int(len(self.splits['train']) * 0.2)
            samples_to_move = self.splits['train'][-num_samples_to_move:]
            self.splits['val'].extend(samples_to_move)
            self.splits['train'] = self.splits['train'][:-num_samples_to_move]
            print(f"Dataset Len: {len(self.splits['train']), len(self.splits['val'])}\n\n\n\n")
        elif self.cfg.split == "splits_MGL_soma_70k_mappred_random.json":
            for stage, names in self.splits.items():
                print("Length of splits {}: ".format(stage), len(self.splits[stage]))
        for stage, names in self.splits.items():
            view = self.dumps[names[0][0]][names[0][1]]["views"][names[0][2]]
            data = {k: [] for k in view.keys() - exclude}
            for scene, seq, name in names:
                for k in data:
                    data[k].append(self.dumps[scene][seq]["views"][name].get(k, None))
            for k in data:
                v = np.array(data[k])
                if np.issubdtype(v.dtype, np.integer) or np.issubdtype(
                    v.dtype, np.floating
                ):
                    v = torch.from_numpy(v)
                data[k] = v
            data["cameras"] = cameras
            data["points"] = points
            self.data[stage] = data
            self.splits[stage] = np.array(names)

    def filter_elements(self):
        for stage, names in self.splits.items():
            names_select = []
            for scene, seq, name in names:
                view = self.dumps[scene][seq]["views"][name]
                if self.cfg.filter_for == "ground_plane":
                    if not (1.0 <= view["height"] <= 3.0):
                        continue
                    planes = self.dumps[scene][seq].get("plane")
                    if planes is not None:
                        inliers = planes[str(view["chunk_id"])][-1]
                        if inliers < 10:
                            continue
                    if self.cfg.filter_by_ground_angle is not None:
                        plane = np.array(view["plane_params"])
                        normal = plane[:3] / np.linalg.norm(plane[:3])
                        angle = np.rad2deg(np.arccos(np.abs(normal[-1])))
                        if angle > self.cfg.filter_by_ground_angle:
                            continue
                elif self.cfg.filter_for == "pointcloud":
                    if len(view["observations"]) < self.cfg.min_num_points:
                        continue
                elif self.cfg.filter_for is not None:
                    raise ValueError(f"Unknown filtering: {self.cfg.filter_for}")
                names_select.append((scene, seq, name))
            logger.info(
                "%s: Keep %d/%d images after filtering for %s.",
                stage,
                len(names_select),
                len(names),
                self.cfg.filter_for,
            )
            self.splits[stage] = names_select

    def parse_splits(self, split_arg, names):
        if split_arg is None:
            self.splits = {
                "train": names,
                "val": names,
            }
        elif isinstance(split_arg, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
        elif isinstance(split_arg, float):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[int(split_arg * len(names)) :],
                "val": names[: int(split_arg * len(names))],
            }
        elif isinstance(split_arg, DictConfig):
            scenes_val = set(split_arg.val)
            scenes_train = set(split_arg.train)
            assert len(scenes_val - set(self.cfg.scenes)) == 0
            assert len(scenes_train - set(self.cfg.scenes)) == 0
            self.splits = {
                "train": [n for n in names if n[0] in scenes_train],
                "val": [n for n in names if n[0] in scenes_val],
            }
        elif isinstance(split_arg, str):
            
            if "/" in split_arg:
                split_path = self.root / split_arg
            else:
                split_path = Path(split_arg)
            
            with split_path.open("r") as fp:
                splits = json.load(fp)
            splits = {
                k: {loc: set(ids) for loc, ids in split.items()}
                for k, split in splits.items()
            }
            self.splits = {}
            
            for k, split in splits.items():
                self.splits[k] = [
                    n
                    for n in names
                    if n[0] in split and int(n[-1].rsplit("_", 1)[0]) in split[n[0]]
                ]
        else:
            raise ValueError(split_arg)

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.splits[stage],
            self.data[stage],
            self.image_dirs,
            self.seg_masks_dir,
            self.flood_masks_dir,

            image_ext=".jpg",
        )

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.splits[stage]
        seq2indices = defaultdict(list)
        for index, (_, seq, _) in enumerate(keys):
            seq2indices[seq].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(self.data[stage], indices, **kwargs)
            for i, sub_indices in enumerate(chunks):
                chunk2indices[seq, i] = sub_indices
        # store the index of each chunk in its sequence
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        dataset = self.dataset(stage)
        return dataset, chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        chunk_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(chunk_keys))
            chunk_keys = [chunk_keys[i] for i in perm]
        key_indices = [i for key in chunk_keys for i in chunk2idx[key]]
        num_workers = self.cfg.loading[stage]["num_workers"]
        loader = torchdata.DataLoader(
            dataset,
            batch_size=None,
            sampler=key_indices,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, chunk_keys, chunk2idx
