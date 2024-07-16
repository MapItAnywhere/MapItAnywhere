# Adapted from PixLoc, Paul-Edouard Sarlin, ETH Zurich
# https://github.com/cvg/pixloc
# Released under the Apache License 2.0
import os, sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from pathlib import Path
import logging
from typing import Any
from dataclasses import dataclass
from .models.schema import ModelConfiguration
from hydra.core.config_store import ConfigStore
import pytorch_lightning  # noqa: F401


formatter = logging.Formatter(
    fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

logger = logging.getLogger("mapper")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

pl_logger = logging.getLogger("pytorch_lightning")
if len(pl_logger.handlers):
    pl_logger.handlers[0].setFormatter(formatter)

repo_dir = Path(__file__).parent.parent
EXPERIMENTS_PATH = repo_dir / "experiments/"
DATASETS_PATH = repo_dir / "datasets/"


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

