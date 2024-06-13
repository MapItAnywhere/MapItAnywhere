from ..base import DataBase
from .dataset import NuScenesDataset
from ..schema import NuScenesDataConfiguration

class NuScenesData(DataBase):
    def __init__(self, cfg: NuScenesDataConfiguration):
        self.cfg = cfg
        self._dataset = {}

    def prepare_data(self):
        pass

    def setup(self, stage):
        if stage is None:
            stage = 'fit'

        split = {
            'fit': 'train',
            'val': 'val',
            'validate': 'val',
            'test': 'test'
        }[stage]

        self._dataset[split] = NuScenesDataset(
            split=split,
            cfg=self.cfg
        )

    def dataset(self, stage):
        if self._dataset.get(stage) is None:
            self.setup(stage)

        return self._dataset[stage]