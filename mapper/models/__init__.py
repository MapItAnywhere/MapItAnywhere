# Adapted from PixLoc, Paul-Edouard Sarlin, ETH Zurich
# https://github.com/cvg/pixloc
# Released under the Apache License 2.0

import inspect
from hydra.core.config_store import ConfigStore

from .schema import DINOConfiguration, ResNetConfiguration
from .base import BaseModel


def get_class(mod_name, base_path, BaseClass):
    """Get the class object which inherits from BaseClass and is defined in
    the module named mod_name, child of base_path.
    """
    mod_path = "{}.{}".format(base_path, mod_name)
    mod = __import__(mod_path, fromlist=[""])
    classes = inspect.getmembers(mod, inspect.isclass)
    # Filter classes defined in the module
    classes = [c for c in classes if c[1].__module__ == mod_path]
    # Filter classes inherited from BaseModel
    classes = [c for c in classes if issubclass(c[1], BaseClass)]
    assert len(classes) == 1, classes
    return classes[0][1]


def get_model(name):
    return get_class(name, __name__, BaseModel)

cs = ConfigStore.instance()
cs.store(group="model/schema/backbone", name="dino", node=DINOConfiguration, package="model.image_encoder.backbone")
cs.store(group="model/schema/backbone", name="resnet", node=ResNetConfiguration, package="model.image_encoder.backbone")