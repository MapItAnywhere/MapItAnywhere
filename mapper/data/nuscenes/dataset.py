import os
import torch
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from itertools import chain
from PIL import Image
from torchvision import transforms as T
import torchvision.transforms as tvf
from torchvision.transforms.functional import to_tensor

from .splits_roddick import create_splits_scenes_roddick
from ..image import pad_image, rectify_image, resize_image
from .utils import decode_binary_labels
from ..utils import decompose_rotmat
from ...utils.io import read_image
from ...utils.wrappers import Camera
from ..schema import NuScenesDataConfiguration


class NuScenesDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: NuScenesDataConfiguration, split="train"):

        self.cfg = cfg
        self.nusc = NuScenes(version=cfg.version, dataroot=str(cfg.data_dir))
        self.map_data_root = cfg.map_dir
        self.split = split

        self.scenes = create_splits_scenes_roddick() # custom based on Roddick et al. 

        scene_split = {
            'v1.0-trainval': {'train': 'train', 'val': 'val', 'test': 'val'},
            'v1.0-mini': {'train': 'mini_train', 'val': 'mini_val'},
        }[cfg.version][split]
        self.scenes = self.scenes[scene_split]
        self.sample = list(filter(lambda sample: self.nusc.get(
            'scene', sample['scene_token'])['name'] in self.scenes, self.nusc.sample))

        self.tfs = self.get_augmentations() if split == "train" else T.Compose([])

        data_tokens = []
        for sample in self.sample:
            data_token = sample['data']
            data_token = [v for k,v in data_token.items() if k == "CAM_FRONT"]

            data_tokens.append(data_token)

        data_tokens = list(chain.from_iterable(data_tokens))
        data = [self.nusc.get('sample_data', token) for token in data_tokens]

        self.data = []
        for d in data:
            sample = self.nusc.get('sample', d['sample_token'])
            scene = self.nusc.get('scene', sample['scene_token'])
            location = self.nusc.get('log', scene['log_token'])['location']

            file_name = d['filename']
            ego_pose = self.nusc.get('ego_pose', d['ego_pose_token'])
            calibrated_sensor = self.nusc.get(
                "calibrated_sensor", d['calibrated_sensor_token'])

            ego2global = np.eye(4).astype(np.float32)
            ego2global[:3, :3] = Quaternion(ego_pose['rotation']).rotation_matrix
            ego2global[:3, 3] = ego_pose['translation']

            sensor2ego = np.eye(4).astype(np.float32)
            sensor2ego[:3, :3] = Quaternion(
                calibrated_sensor['rotation']).rotation_matrix
            sensor2ego[:3, 3] = calibrated_sensor['translation']

            sensor2global = ego2global @ sensor2ego

            rotation = sensor2global[:3, :3]
            roll, pitch, yaw = decompose_rotmat(rotation)

            fx = calibrated_sensor['camera_intrinsic'][0][0]
            fy = calibrated_sensor['camera_intrinsic'][1][1]
            cx = calibrated_sensor['camera_intrinsic'][0][2]
            cy = calibrated_sensor['camera_intrinsic'][1][2]
            width = d['width']
            height = d['height']

            cam = Camera(torch.tensor(
                [width, height, fx, fy, cx - 0.5, cy - 0.5])).float()
            self.data.append({
                'filename': file_name,
                'yaw': yaw,
                'pitch': pitch,
                'roll': roll,
                'cam': cam,
                'sensor2global': sensor2global,
                'token': d['token'],
                'sample_token': d['sample_token'],
                'location': location
            })
        
        if self.cfg.percentage < 1.0 and split == "train":
            self.data = self.data[:int(len(self.data) * self.cfg.percentage)]

    def get_augmentations(self):

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]

        image = read_image(os.path.join(self.nusc.dataroot, d['filename']))
        image = np.array(image)
        cam = d['cam']
        roll = d['roll']
        pitch = d['pitch']
        yaw = d['yaw']

        with Image.open(self.map_data_root / f"{d['token']}.png") as semantic_image:
            semantic_mask = to_tensor(semantic_image)

        semantic_mask = decode_binary_labels(semantic_mask, self.cfg.num_classes + 1)
        semantic_mask = torch.nn.functional.max_pool2d(semantic_mask.float(), (2, 2), stride=2) # 2 times downsample
        semantic_mask = semantic_mask.permute(1, 2, 0)
        semantic_mask = torch.flip(semantic_mask, [0])
        
        visibility_mask = semantic_mask[..., -1]
        semantic_mask = semantic_mask[..., :-1]

        if self.cfg.class_mapping is not None:
            semantic_mask = semantic_mask[..., self.cfg.class_mapping]

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
        if self.cfg.resize_image is not None:
            image, _, cam, valid = resize_image(
                image, self.cfg.resize_image, fn=max, camera=cam, valid=valid
            )
            if self.cfg.pad_to_square:
                image, valid, cam = pad_image(image, self.cfg.resize_image, cam, valid)
        image = self.tfs(image)

        confidence_map = visibility_mask.clone().float()
        confidence_map = (confidence_map - confidence_map.min()) / (confidence_map.max() - confidence_map.min())

        return {
            "image": image,
            "roll_pitch_yaw": torch.tensor([roll, pitch, yaw]).float(),
            "camera": cam,
            "valid": valid,
            "seg_masks": semantic_mask.float(),
            "token": d['token'],
            "sample_token": d['sample_token'],
            'location': d['location'],
            'flood_masks': visibility_mask.float(),
            "confidence_map": confidence_map,
            'name': d['sample_token']
        }
