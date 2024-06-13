import time
import torch
import hydra
import pytorch_lightning as pl
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from pathlib import Path
from dataclasses import dataclass

from .module import GenericModule
from .data.module import GenericDataModule
from .callbacks import EvalSaveCallback, ImageLoggerCallback
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


@hydra.main(version_base=None, config_path="conf", config_name="pretrain")
def train(cfg: Configuration):
    OmegaConf.resolve(cfg)

    dm = GenericDataModule(cfg.data)

    model = GenericModule(cfg)

    exp_name_with_time = cfg.experiment.name + \
        "_" + time.strftime("%Y-%m-%d_%H-%M-%S")

    callbacks: list[pl.Callback]

    if cfg.training.eval:
        save_dir = Path(cfg.training.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            EvalSaveCallback(save_dir=save_dir)
        ]

        logger = None
    else:
        callbacks = [
            ImageLoggerCallback(num_classes=cfg.training.num_classes),
            ModelCheckpoint(
                monitor=cfg.training.checkpointing.monitor,
                save_last=cfg.training.checkpointing.save_last,
                save_top_k=cfg.training.checkpointing.save_top_k,
            )
        ]

        logger = WandbLogger(
            name=exp_name_with_time,
            id=exp_name_with_time,
            entity="mappred-large",
            project="map-pred-full-v3",
        )

        logger.watch(model, log="all", log_freq=500)

    if cfg.training.checkpoint is not None:
        state_dict = torch.load(cfg.training.checkpoint)['state_dict']
        model.load_state_dict(state_dict, strict=False)

    trainer_args = OmegaConf.to_container(cfg.training.trainer)
    trainer_args['callbacks'] = callbacks
    trainer_args['logger'] = logger

    trainer = pl.Trainer(**trainer_args)

    if cfg.training.eval:
        trainer.test(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("high")

    train()
