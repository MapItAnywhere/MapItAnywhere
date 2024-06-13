# Copyright (c) Meta Platforms, Inc. and affiliates.

# Adapted from PixLoc, Paul-Edouard Sarlin, ETH Zurich
# https://github.com/cvg/pixloc
# Released under the Apache License 2.0

"""
Base class for trainable models.
"""

from abc import ABCMeta, abstractmethod
from copy import copy

from omegaconf import OmegaConf
from torch import nn


class BaseModel(nn.Module, metaclass=ABCMeta):

    required_data_keys = []
    strict_conf = True

    def __init__(self, conf):
        """Perform some logic and call the _init method of the child model."""
        super().__init__()
        self.conf = conf 
        OmegaConf.set_readonly(conf, True)
        OmegaConf.set_struct(conf, True)
        self.required_data_keys = copy(self.required_data_keys)
        self._init(conf)

    def forward(self, data):
        """Check the data and call the _forward method of the child model."""

        def recursive_key_check(expected, given):
            for key in expected:
                assert key in given, f"Missing key {key} in data"
                if isinstance(expected, dict):
                    recursive_key_check(expected[key], given[key])

        recursive_key_check(self.required_data_keys, data)
        return self._forward(data)

    @abstractmethod
    def _init(self, conf):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def _forward(self, data):
        """To be implemented by the child class."""
        raise NotImplementedError

    def loss(self, pred, data):
        """To be implemented by the child class."""
        raise NotImplementedError

    def metrics(self):
        return {}  # no metrics
