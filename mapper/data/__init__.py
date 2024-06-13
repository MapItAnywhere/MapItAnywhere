from .mapillary.data_module import MapillaryDataModule
from .nuscenes.data_module import NuScenesData

modules = {
    "mapillary": MapillaryDataModule,
    "nuscenes": NuScenesData
}
