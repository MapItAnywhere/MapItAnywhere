from ..base import DataBase
from .dataset import BEVKitti360Dataset
from ..schema import KITTIDataConfiguration

class BEVKitti360Data(DataBase):
    def __init__(self, cfg: KITTIDataConfiguration) -> None:
        self.cfg = cfg
        self._dataset = {}

    def prepare_data(self) -> None:
        return
    
    def setup(self, stage: str) -> None:
        split = {
            'fit': 'train',
            'val': 'val',
            'validate': 'val',
            'test': 'val',
            "train": "train"
        }[stage]

        self._dataset[stage] = BEVKitti360Dataset(
            cfg=self.cfg,
            split_name=split
        )

    def dataset(self, stage: str):
        if self._dataset.get(stage) is None:
            self.setup(stage)
        
        return self._dataset[stage]
    