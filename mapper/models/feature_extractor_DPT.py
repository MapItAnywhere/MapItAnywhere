from .base import BaseModel
from .schema import DINOConfiguration
import logging
import torch
import torch.nn as nn

import sys
import re
import os

from .dinov2.eval.depth.ops.wrappers import resize
from .dinov2.hub.backbones import dinov2_vitb14_reg

module_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_dir)

logger = logging.getLogger(__name__)


class FeatureExtractor(BaseModel):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def build_encoder(self, conf: DINOConfiguration):
        BACKBONE_SIZE = "small"
        backbone_archs = {
            "small": "vits14",
            "base": "vitb14",  # this one
            "large": "vitl14",
            "giant": "vitg14",
        }
        backbone_arch = backbone_archs[BACKBONE_SIZE]
        self.crop_size = int(re.search(r"\d+", backbone_arch).group())
        backbone_name = f"dinov2_{backbone_arch}"

        self.backbone_model = dinov2_vitb14_reg(
            pretrained=conf.pretrained, drop_path_rate=0.1)

        if conf.frozen:
            for param in self.backbone_model.patch_embed.parameters():
                param.requires_grad = False

            for i in range(0, 10):
                for param in self.backbone_model.blocks[i].parameters():
                    param.requires_grad = False
                self.backbone_model.blocks[i].drop_path1 = nn.Identity()
                self.backbone_model.blocks[i].drop_path2 = nn.Identity()

        self.feat_projection = torch.nn.Conv2d(
            768, conf.output_dim, kernel_size=1)

        return self.backbone_model

    def _init(self, conf: DINOConfiguration):
        # Preprocessing
        self.register_buffer("mean_", torch.tensor(
            self.mean), persistent=False)
        self.register_buffer("std_", torch.tensor(self.std), persistent=False)

        self.build_encoder(conf)

    def _forward(self, data):
        _, _, h, w = data["image"].shape

        h_num_patches = h // self.crop_size
        w_num_patches = w // self.crop_size

        h_dino = h_num_patches * self.crop_size
        w_dino = w_num_patches * self.crop_size

        image = resize(data["image"], (h_dino, w_dino))

        image = (image - self.mean_[:, None, None]) / self.std_[:, None, None]

        output = self.backbone_model.forward_features(
            image)['x_norm_patchtokens']
        output = output.reshape(-1, h_num_patches,
                                w_num_patches, output.shape[-1])
        output = output.permute(0, 3, 1, 2)  # channel first
        output = self.feat_projection(output)

        camera = data['camera'].to(data["image"].device, non_blocking=True)
        camera = camera.scale(output.shape[-1] / data["image"].shape[-1])

        return output, camera
