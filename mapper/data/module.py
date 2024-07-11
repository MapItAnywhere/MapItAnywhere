# Adapted from OrienterNet, Paul-Edouard Sarlin, ETH Zurich
# https://github.com/facebookresearch/OrienterNet
# Released under the CC-BY-NC license

from typing import Optional
from omegaconf import DictConfig
import pytorch_lightning as L
import torch.utils.data as torchdata
from .torch import collate, worker_init_fn


def get_dataset(name):
    if name == "mapillary":
        from .mapillary.data_module import MapillaryDataModule
        return MapillaryDataModule
    elif name == "nuscenes":
        from .nuscenes.data_module import NuScenesData
        return NuScenesData
    elif name == "kitti":
        from .kitti.data_module import BEVKitti360Data
        return BEVKitti360Data
    else:
        raise NotImplementedError(f"Dataset {name} not implemented.")


class GenericDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.data_module = get_dataset(cfg.name)(cfg)

    def prepare_data(self) -> None:
        self.data_module.prepare_data()
    
    def setup(self, stage: Optional[str] = None):
        self.data_module.setup(stage)

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.data_module.dataset(stage)
        cfg = self.cfg["loading"][stage]
        num_workers = cfg["num_workers"] if num_workers is None else num_workers
        loader = torchdata.DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=num_workers,
            shuffle=shuffle or (stage == "train"),
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader
    
    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)