from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class LossConfiguration:
    num_classes: int

    xent_weight: float = 1.0
    dice_weight: float = 1.0
    focal_loss: bool = False
    focal_loss_gamma: float = 2.0
    requires_frustrum: bool = True
    requires_flood_mask: bool = False
    class_weights: Optional[Any] = None
    label_smoothing: float = 0.1

@dataclass
class BackboneConfigurationBase:
    pretrained: bool
    frozen: bool
    output_dim: int

@dataclass
class DINOConfiguration(BackboneConfigurationBase):
    pretrained: bool = True
    frozen: bool = False
    output_dim: int = 128

@dataclass
class ResNetConfiguration(BackboneConfigurationBase):
    input_dim: int
    encoder: str
    remove_stride_from_first_conv: bool
    num_downsample: Optional[int]
    decoder_norm: str
    do_average_pooling: bool
    checkpointed: bool

@dataclass
class ImageEncoderConfiguration:
    name: str
    backbone: Any

@dataclass
class ModelConfiguration:
    segmentation_head: Dict[str, Any]
    image_encoder: ImageEncoderConfiguration

    name: str
    num_classes: int
    latent_dim: int
    z_max: int
    x_max: int
    
    pixel_per_meter: int
    num_scale_bins: int

    loss: LossConfiguration

    scale_range: list[int] = field(default_factory=lambda: [0, 9])
    z_min: Optional[int] = None