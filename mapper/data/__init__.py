from .mapillary.data_module import MapillaryDataModule
from .nuscenes.data_module import NuScenesData
from hydra.core.config_store import ConfigStore
from .schema import MIADataConfiguration, KITTIDataConfiguration, NuScenesDataConfiguration

modules = {
    "mapillary": MapillaryDataModule,
    "nuscenes": NuScenesData
}

cs = ConfigStore.instance()

cs.store(group="schema/data", name="mia",
         node=MIADataConfiguration, package="data")
cs.store(group="schema/data", name="kitti", node=KITTIDataConfiguration, package="data")
cs.store(group="schema/data", name="nuscenes", node=NuScenesDataConfiguration, package="data")
