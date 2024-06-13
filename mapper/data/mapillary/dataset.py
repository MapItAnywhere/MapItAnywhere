from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.utils.data as torchdata
import torchvision.transforms as tvf
from PIL import Image
from pathlib import Path

from ...models.utils import deg2rad, rotmat2d
from ...utils.io import read_image
from ...utils.wrappers import Camera
from ..image import pad_image, rectify_image, resize_image
from ..utils import decompose_rotmat
from ..schema import MIADataConfiguration


class MapLocDataset(torchdata.Dataset):
    def __init__(
        self,
        stage: str,
        cfg: MIADataConfiguration,
        names: List[str],
        data: Dict[str, Any],
        image_dirs: Dict[str, Path],
        seg_mask_dirs: Dict[str, Path],
        flood_masks_dirs: Dict[str, Path],
        image_ext: str = "",
    ):
        self.stage = stage
        self.cfg = deepcopy(cfg)
        self.data = data
        self.image_dirs = image_dirs
        self.seg_mask_dirs = seg_mask_dirs
        self.flood_masks_dirs = flood_masks_dirs
        self.names = names
        self.image_ext = image_ext

        tfs = []
        self.tfs = tvf.Compose(tfs)
        self.augmentations = self.get_augmentations()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if self.stage == "train" and self.cfg.random:
            seed = None
        else:
            seed = [self.cfg.seed, idx]
        (seed,) = np.random.SeedSequence(seed).generate_state(1)

        scene, seq, name = self.names[idx]

        view = self.get_view(
            idx, scene, seq, name, seed
        )

        return view

    def get_augmentations(self):
        if self.stage != "train" or not self.cfg.augmentations.enabled:
            print(f"No Augmentation!", "\n" * 10)
            self.cfg.augmentations.random_flip = 0.0
            return tvf.Compose([])

        print(f"Augmentation!", "\n" * 10)
        augmentations = [
            tvf.ColorJitter(
                brightness=self.cfg.augmentations.brightness,
                contrast=self.cfg.augmentations.contrast,
                saturation=self.cfg.augmentations.saturation,
                hue=self.cfg.augmentations.hue,
            )
        ]

        if self.cfg.augmentations.random_resized_crop:
            augmentations.append(
                tvf.RandomResizedCrop(scale=(0.8, 1.0))
            )  # RandomResizedCrop

        if self.cfg.augmentations.gaussian_noise.enabled:
            augmentations.append(
                tvf.GaussianNoise(
                    mean=self.cfg.augmentations.gaussian_noise.mean,
                    std=self.cfg.augmentations.gaussian_noise.std,
                )
            )  # Gaussian noise

        if self.cfg.augmentations.brightness_contrast.enabled:
            augmentations.append(
                tvf.ColorJitter(
                    brightness=self.cfg.augmentations.brightness_contrast.brightness_factor,
                    contrast=self.cfg.augmentations.brightness_contrast.contrast_factor,
                    saturation=0,  # Keep saturation at 0 for brightness and contrast adjustment
                    hue=0,
                )
            )  # Brightness and contrast adjustment

        return tvf.Compose(augmentations)

    def random_flip(self, image, cam, valid, seg_mask, flood_mask, conf_mask):
        if torch.rand(1) < self.cfg.augmentations.random_flip:
            image = torch.flip(image, [-1])
            cam = cam.flip()
            valid = torch.flip(valid, [-1])
            seg_mask = torch.flip(seg_mask, [1])
            flood_mask = torch.flip(flood_mask, [-1])
            conf_mask = torch.flip(conf_mask, [-1])

        return image, cam, valid, seg_mask, flood_mask, conf_mask

    def get_view(self, idx, scene, seq, name, seed):
        data = {
            "index": idx,
            "name": name,
            "scene": scene,
            "sequence": seq,
        }
        cam_dict = self.data["cameras"][scene][seq][self.data["camera_id"][idx]]
        cam = Camera.from_dict(cam_dict).float()

        if "roll_pitch_yaw" in self.data:
            roll, pitch, yaw = self.data["roll_pitch_yaw"][idx].numpy()
        else:
            roll, pitch, yaw = decompose_rotmat(
                self.data["R_c2w"][idx].numpy())

        image = read_image(self.image_dirs[scene] / (name + self.image_ext))
        image = Image.fromarray(image)
        image = self.augmentations(image)
        image = np.array(image)

        if "plane_params" in self.data:
            # transform the plane parameters from world to camera frames
            plane_w = self.data["plane_params"][idx]
            data["ground_plane"] = torch.cat(
                [rotmat2d(deg2rad(torch.tensor(yaw)))
                 @ plane_w[:2], plane_w[2:]]
            )

        image, valid, cam, roll, pitch = self.process_image(
            image, cam, roll, pitch, seed
        )

        if "chunk_index" in self.data:  # TODO: (cherie) do we need this?
            data["chunk_id"] = (scene, seq, self.data["chunk_index"][idx])

        # Semantic map extraction
        seg_mask_path = self.seg_mask_dirs[scene] / \
            (name.split("_")[0] + ".npy")
        seg_masks_ours = np.load(seg_mask_path)
        mask_center = (
            seg_masks_ours.shape[0] // 2, seg_masks_ours.shape[1] // 2)

        seg_masks_ours = seg_masks_ours[mask_center[0] -
                                        100:mask_center[0], mask_center[1] - 50: mask_center[1] + 50]

        if self.cfg.num_classes == 6:
            seg_masks_ours = seg_masks_ours[..., [0, 1, 2, 4, 6, 7]]

        flood_mask_path = self.flood_masks_dirs[scene] / \
            (name.split("_")[0] + ".npy")
        flood_mask = np.load(flood_mask_path)

        flood_mask = flood_mask[mask_center[0]-100:mask_center[0],
                                mask_center[1] - 50: mask_center[1] + 50]

        confidence_map = flood_mask.copy()
        confidence_map = (confidence_map - confidence_map.min()) / \
            (confidence_map.max() - confidence_map.min() + 1e-6)

        seg_masks_ours = torch.from_numpy(seg_masks_ours).float()
        flood_mask = torch.from_numpy(flood_mask).float()
        confidence_map = torch.from_numpy(confidence_map).float()

        # Map Augmentations
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            image, cam, valid, seg_masks_ours, flood_mask, confidence_map = self.random_flip(
                image, cam, valid, seg_masks_ours, flood_mask, confidence_map)

        return {
            **data,
            "image": image,
            "valid": valid,
            "camera": cam,
            "seg_masks": seg_masks_ours,
            "flood_masks": flood_mask,
            "roll_pitch_yaw": torch.tensor((roll, pitch, yaw)).float(),
            "confidence_map": confidence_map
            # "pixels_per_meter": torch.tensor(canvas.ppm).float(),
        }

    def process_image(self, image, cam, roll, pitch, seed):
        image = (
            torch.from_numpy(np.ascontiguousarray(image))
            .permute(2, 0, 1)
            .float()
            .div_(255)
        )

        if not self.cfg.gravity_align:
            # Turn off gravity alignment
            roll = 0.0
            pitch = 0.0
            image, valid = rectify_image(image, cam, roll, pitch)
        else:
            image, valid = rectify_image(
                image, cam, roll, pitch if self.cfg.rectify_pitch else None
            )
            roll = 0.0
            if self.cfg.rectify_pitch:
                pitch = 0.0

        if self.cfg.target_focal_length is not None:
            # Resize to a canonical focal length
            factor = self.cfg.target_focal_length / cam.f.numpy()
            size = (np.array(image.shape[-2:][::-1]) * factor).astype(int)
            image, _, cam, valid = resize_image(
                image, size, camera=cam, valid=valid)
            size_out = self.cfg.resize_image
            if size_out is None:
                # Round the edges up such that they are multiple of a factor
                stride = self.cfg.pad_to_multiple
                size_out = (np.ceil((size / stride)) * stride).astype(int)
            # Crop or pad such that both edges are of the given size
            image, valid, cam = pad_image(
                image, size_out, cam, valid, crop_and_center=True
            )
        elif self.cfg.resize_image is not None:
            image, _, cam, valid = resize_image(
                image, self.cfg.resize_image, fn=max, camera=cam, valid=valid
            )
            if self.cfg.pad_to_square:
                # Pad such that both edges are of the given size
                image, valid, cam = pad_image(
                    image, self.cfg.resize_image, cam, valid)

        if self.cfg.reduce_fov is not None:
            h, w = image.shape[-2:]
            f = float(cam.f[0])
            fov = np.arctan(w / f / 2)
            w_new = round(2 * f * np.tan(self.cfg.reduce_fov * fov))
            image, valid, cam = pad_image(
                image, (w_new, h), cam, valid, crop_and_center=True
            )

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            image = self.tfs(image)

        return image, valid, cam, roll, pitch
