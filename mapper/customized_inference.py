from matplotlib import pyplot as plt
from mapper.utils.io import read_image
from mapper.utils.exif import EXIF
from mapper.utils.wrappers import Camera
from mapper.data.image import rectify_image, resize_image
from mapper.utils.viz_2d import one_hot_argmax_to_rgb, plot_images
from mapper.module import GenericModule
from perspective2d import PerspectiveFields
import torch
import numpy as np
from typing import Optional, Tuple
import glob 
import hydra
from hydra.core.config_store import ConfigStore
from typing import Any
from dataclasses import dataclass

from .models.schema import ModelConfiguration, DINOConfiguration, ResNetConfiguration
from .data.schema import MIADataConfiguration, KITTIDataConfiguration, NuScenesDataConfiguration

@dataclass
class ExperimentConfiguration:
    name: str

@dataclass
class Configuration:
    model: ModelConfiguration
    experiment: ExperimentConfiguration
    data: Any
    training: Any

    image_path: str
    save_path: str = "output.png"


cs = ConfigStore.instance()

# Store root configuration schema
cs.store(name="pretrain", node=Configuration)
cs.store(name="mapper_nuscenes", node=Configuration)
cs.store(name="mapper_kitti", node=Configuration)

# Store data configuration schema
cs.store(group="schema/data", name="mia",
         node=MIADataConfiguration, package="data")
cs.store(group="schema/data", name="kitti", node=KITTIDataConfiguration, package="data")
cs.store(group="schema/data", name="nuscenes", node=NuScenesDataConfiguration, package="data")

cs.store(group="model/schema/backbone", name="dino", node=DINOConfiguration, package="model.image_encoder.backbone")
cs.store(group="model/schema/backbone", name="resnet", node=ResNetConfiguration, package="model.image_encoder.backbone")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageCalibrator(PerspectiveFields):
    def __init__(self, version: str = "Paramnet-360Cities-edina-centered"):
        super().__init__(version)
        self.eval()

    def run(
        self,
        image_rgb: np.ndarray,
        focal_length: Optional[float] = None,
        exif: Optional[EXIF] = None,
    ) -> Tuple[Tuple[float, float], Camera]:
        h, w, *_ = image_rgb.shape
        if focal_length is None and exif is not None:
            _, focal_ratio = exif.extract_focal()
            if focal_ratio != 0:
                focal_length = focal_ratio * max(h, w)
        calib = self.inference(img_bgr=image_rgb[..., ::-1])
        roll_pitch = (calib["pred_roll"].item(), calib["pred_pitch"].item())
        if focal_length is None:
            vfov = calib["pred_vfov"].item()
            focal_length = h / 2 / np.tan(np.deg2rad(vfov) / 2)
        camera = Camera.from_dict(
            {
                "model": "SIMPLE_PINHOLE",
                "width": w,
                "height": h,
                "params": [focal_length, w / 2 + 0.5, h / 2 + 0.5],
            }
        )
        return roll_pitch, camera

def preprocess_pipeline(image, roll_pitch, camera):
    image = torch.from_numpy(image).float() / 255
    image = image.permute(2, 0, 1).to(device)
    camera = camera.to(device)

    image, valid = rectify_image(image, camera.float(), -roll_pitch[0], -roll_pitch[1])

    roll_pitch *= 0

    image, _, camera, valid = resize_image(
        image=image,
        size=512,
        camera=camera,
        fn=max,
        valid=valid
    )

    camera = torch.stack([camera])

    return {
        "image": image.unsqueeze(0).to(device),
        "valid": valid.unsqueeze(0).to(device),
        "camera": camera.float().to(device),
    }


def infer(calibrator, model, image_path: str):

    image = read_image(image_path)
    with open(image_path, "rb") as fid:
        exif = EXIF(fid, lambda: image.shape[:2])
    
    gravity, camera = calibrator.run(image, exif=exif)

    data = preprocess_pipeline(image, gravity, camera)
    res = model(data)
    
    prediction = res['output']
    rgb_prediction = one_hot_argmax_to_rgb(prediction, 6).squeeze(0).permute(1, 2, 0).cpu().long().numpy()
    valid = res['valid_bev'].squeeze(0)[..., :-1]
    rgb_prediction[~valid.cpu().numpy()] = 255
    
    plot_images([image, rgb_prediction], titles=["Input Image", "Top-Down Prediction"], pad=2, adaptive=True)

    return plt.gcf()

@hydra.main(version_base=None, config_path="conf", config_name="pretrain")
def main(cfg: Configuration):
    calibrator = ImageCalibrator().to(device)

    model = GenericModule(cfg)
    state_dict = torch.load(cfg.training.checkpoint, map_location=device)
    model.load_state_dict(state_dict["state_dict"], strict=False)
    model = model.to(device)
    model = model.eval()

    fig = infer(calibrator, model, cfg.image_path)
    fig.savefig(cfg.save_path)

if __name__ == "__main__":
    main()