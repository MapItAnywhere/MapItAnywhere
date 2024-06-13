import numpy as np
import torch
from torchvision.transforms import functional as tfn
import torchvision.transforms.functional as tvf

from ..utils import decompose_rotmat
from ..image import pad_image, rectify_image, resize_image
from ...utils.wrappers import Camera
from ..schema import KITTIDataConfiguration


class BEVTransform:
    def __init__(self,
                 cfg: KITTIDataConfiguration, augmentations):
        self.cfg = cfg
        self.augmentations = augmentations

    @staticmethod
    def _compact_labels(msk, cat, iscrowd):
        ids = np.unique(msk)
        if 0 not in ids:
            ids = np.concatenate((np.array([0], dtype=np.int32), ids), axis=0)

        ids_to_compact = np.zeros((ids.max() + 1,), dtype=np.int32)
        ids_to_compact[ids] = np.arange(0, ids.size, dtype=np.int32)

        msk = ids_to_compact[msk]
        cat = cat[ids]
        iscrowd = iscrowd[ids]

        return msk, cat, iscrowd

    def __call__(self, img, bev_msk=None, bev_plabel=None, fv_msk=None, bev_weights_msk=None,
                 bev_cat=None, bev_iscrowd=None, fv_cat=None, fv_iscrowd=None,
                 fv_intrinsics=None, ego_pose=None):
        # Wrap in np.array
        if bev_cat is not None:
            bev_cat = np.array(bev_cat, dtype=np.int32)
        if bev_iscrowd is not None:
            bev_iscrowd = np.array(bev_iscrowd, dtype=np.uint8)

        if ego_pose is not None:
            ego_pose = np.array(ego_pose, dtype=np.float32)

        roll, pitch, yaw = decompose_rotmat(ego_pose[:3, :3])

        # Image transformations
        img = tfn.to_tensor(img)
        # img = [self._normalize_image(rgb) for rgb in img]
        fx = fv_intrinsics[0][0]
        fy = fv_intrinsics[1][1]
        cx = fv_intrinsics[0][2]
        cy = fv_intrinsics[1][2]
        width = img.shape[2]
        height = img.shape[1]

        cam = Camera(torch.tensor(
                        [width, height, fx, fy, cx - 0.5, cy - 0.5])).float()

        if not self.cfg.gravity_align:
            # Turn off gravity alignment
            roll = 0.0
            pitch = 0.0
            img, valid = rectify_image(img, cam, roll, pitch)
        else:
            img, valid = rectify_image(
                img, cam, roll, pitch if self.cfg.rectify_pitch else None
            )
            roll = 0.0
            if self.cfg.rectify_pitch:
                pitch = 0.0

        if self.cfg.target_focal_length is not None:
            # Resize to a canonical focal length
            factor = self.cfg.target_focal_length / cam.f.numpy()
            size = (np.array(img.shape[-2:][::-1]) * factor).astype(int)
            img, _, cam, valid = resize_image(img, size, camera=cam, valid=valid)
            size_out = self.cfg.resize_image
            if size_out is None:
                # Round the edges up such that they are multiple of a factor
                stride = self.cfg.pad_to_multiple
                size_out = (np.ceil((size / stride)) * stride).astype(int)
            # Crop or pad such that both edges are of the given size
            img, valid, cam = pad_image(
                img, size_out, cam, valid, crop_and_center=False
            )
        elif self.cfg.resize_image is not None:
            img, _, cam, valid = resize_image(
                img, self.cfg.resize_image, fn=max, camera=cam, valid=valid
            )
            if self.cfg.pad_to_square:
                # Pad such that both edges are of the given size
                img, valid, cam = pad_image(img, self.cfg.resize_image, cam, valid)

        # Label transformations,
        if bev_msk is not None:
            bev_msk = np.expand_dims(
                np.array(bev_msk, dtype=np.int32, copy=False),
                axis=0
            )
            bev_msk, bev_cat, bev_iscrowd = self._compact_labels(
                bev_msk, bev_cat, bev_iscrowd
            )

            bev_msk = torch.from_numpy(bev_msk)
            bev_cat = torch.from_numpy(bev_cat)

            rotated_mask = torch.rot90(bev_msk, dims=(1, 2))
            cropped_mask = rotated_mask[:, :672, (rotated_mask.size(2) - 672) // 2:-(rotated_mask.size(2) - 672) // 2]

            bev_msk = cropped_mask.squeeze(0)
            seg_masks = bev_cat[bev_msk]

            seg_masks_onehot = seg_masks.clone()
            seg_masks_onehot[seg_masks_onehot == 255] = 0
            seg_masks_onehot = torch.nn.functional.one_hot(
                seg_masks_onehot.to(torch.int64),
                num_classes=self.cfg.num_classes
            )
            seg_masks_onehot[seg_masks == 255] = 0

            seg_masks_onehot = seg_masks_onehot.permute(2, 0, 1)

            seg_masks_down = tvf.resize(seg_masks_onehot, (100, 100))

            seg_masks_down = seg_masks_down.permute(1, 2, 0)

            if self.cfg.class_mapping is not None:
                seg_masks_down = seg_masks_down[:, :, self.cfg.class_mapping]

        img = self.augmentations(img)
        flood_masks = torch.all(seg_masks_down == 0, dim=2).float()


        ret = {
            "image": img,
            "valid": valid,
            "camera": cam,
            "seg_masks": (seg_masks_down).float().contiguous(),
            "flood_masks": flood_masks,
            "roll_pitch_yaw": torch.tensor((roll, pitch, yaw)).float(),
            "confidence_map": flood_masks,
        }
        
        for key, value in ret.items():
            if isinstance(value, np.ndarray):
                ret[key] = torch.from_numpy(value)
    
        return ret
