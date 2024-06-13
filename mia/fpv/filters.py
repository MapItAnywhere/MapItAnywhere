"""
Contains the filters used to filter out images from the Mapillary API.
"""

import inspect
import yaml
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import shapely
import shapely.geometry
from shapely.prepared import prep
from shapely import contains_xy

from .. import logger

def in_shape_filter(df: pd.DataFrame, geojson_shape):
    polygon = shapely.geometry.shape(geojson_shape["features"][0]["geometry"])
    mask = contains_xy(polygon, x=df["geometry.long"], y=df["geometry.lat"])
    return mask

def value_range_filter(df: pd.DataFrame, key, from_v=None, to_v=None):
    c = df[key]
    if from_v is not None and to_v is not None:
        if from_v == to_v:
            return c == from_v
        else:
            return np.logical_and(c >= from_v, c <= to_v)
    elif from_v is not None:
        return c >= from_v
    elif to_v is not None:
        return c <= to_v
    else:
        raise Exception("from_v and to_v cannot both be None")
    
def value_in_list_filter(df: pd.DataFrame, key, lst, exclude=False):
    mask = df[key].isin(lst)
    if exclude:
        mask = ~mask
    return mask


def value_missing_filter(df: pd.DataFrame, keys):
    return np.all(df[keys].notna(), axis=1)


def date_filter(df: pd.DataFrame, from_year=None, to_year=None):
    """
    Args:
        before_year: integer representing the year
        after_year: integer representing the year
    """
    if from_year is not None:
        from_year = int(datetime(from_year, 1, 1).timestamp())*1e3
    if to_year is not None:
        to_year = int(datetime(to_year, 1, 1).timestamp())*1e3
    return value_range_filter(df, "captured_at", from_year, to_year)

def quality_score_filter(df: pd.DataFrame, from_score=None, to_score=None):
    return value_range_filter(df, "quality_score", from_v=from_score, to_v=to_score)

def angle_dist(a1, a2):
    a = a1-a2
    return np.abs((a + 180) % 360 - 180)

def angle_discrip_filter(df: pd.DataFrame, thresh, less_than=True):
    """
    Args:
        thresh: Threshold in degrees
    """
    a1 = df["computed_compass_angle"]
    a2 = df["compass_angle"]

    diff = angle_dist(a1, a2)

    if less_than:
        return diff < thresh
    else:
        return diff > thresh

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    
    All args must be of equal length.    
    
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6378.137 * c
    return km*1e3

def loc_discrip_filter(df: pd.DataFrame, thresh, less_than=True):
    """
    Args:
        thresh: Threshold in meters
    """
    lat1 = df["computed_geometry.lat"]
    lon1 = df["computed_geometry.long"]
    lat2 = df["geometry.lat"]
    lon2 = df["geometry.long"]
    diff = haversine_np(lon1, lat1, lon2, lat2)
    if less_than:
        return diff < thresh
    else:
        return diff > thresh
    
def sequence_sparsity_filter(df: pd.DataFrame, dist_thresh):
    """
    TODO
    This filter filters out images that are too close to each other within a sequence
    """
    pass
    
    
class Filter():
    def __init__(self, filter_func, name=None, **kwargs):
        self.filter_func = filter_func
        self.name = name
        self.kwargs = kwargs
    
    def __call__(self, df: pd.DataFrame):
        return self.filter_func(df, **self.kwargs)
    
    def __str__(self) -> str:
        if self.name is None:
            tag = self.filter_func.__name__
        else:
            tag = f"{self.filter_func.__name__}:{self.name}"
        return tag
    
    def __repr__(self):
        kwargs_fmt = ", ".join([f"{k}={v}" for k,v in self.kwargs.items()])
        return f"{self.__str__()} | kwargs({kwargs_fmt})"


class FilterPipeline():
    def __init__(self, filters: list, sequential=True, name=None, verbose=True):
        """
        Args:
            sequential: Whether to apply filters sequentially or compute the masks
            for all of them then apply once at the end.
            verbose: Whether to log the effect of each filter or not
        """
        self.filters = filters
        self.sequential = sequential
        self.name = name
        self.verbose = verbose

    def __call__(self, df: pd.DataFrame):
        N = df.shape[0]
        if not self.sequential:
            running_mask = np.full(df.shape[0], True, dtype=bool)

        for f in self.filters:
            mask = f(df)
            if self.verbose:
                s = np.sum(mask)
                logger.info(f"{f} keeps {s}/{mask.shape[0]} ({s/mask.shape[0]*100:.2f}%) of the images")

            if self.sequential:
                df = df[mask]
                if df.shape[0] == 0:
                    logger.warn("No images left during filtering.. Stopping pipeline")
                    return df
            else:
                running_mask = np.logical_and(running_mask, mask)
        
        if not self.sequential:
            df = df[running_mask]
        
        logger.info(f"Filter Pipeline {self.name} kept {df.shape[0]}/{N} ({df.shape[0]/N*100:.2f}%) of the images")
        return df

    def __str__(self):
        return f"Pipeline {self.name}: " + "\n".join([str(x) for x in self.filters])
    
    def __repr__(self):
        return f"Pipeline {self.name}: " + "\n".join([repr(x) for x in self.filters])
    
    @staticmethod
    def load_from_yaml(file_path):
        def is_primitive(x):
            return isinstance(x, (float, int, bool, str))

        with open(file_path, 'r') as stream:
            pipeline_dict = yaml.safe_load(stream)["filter_pipeline"]
        
        sig = inspect.signature(FilterPipeline.__init__)
        init_args = dict()
        for param in sig.parameters.values():
            if param.name in pipeline_dict and is_primitive(pipeline_dict[param.name]):
                init_args[param.name] = pipeline_dict[param.name]
        
        filter_dicts = pipeline_dict["filters"]
        filters = list()

        for filter_dict in filter_dicts:
            filter_func_name, kwargs = list(filter_dict.items())[0]
            filter_func = globals()[filter_func_name]
            filters.append(Filter(filter_func=filter_func, **kwargs))

        pipeline = FilterPipeline(filters, **init_args)
        return pipeline

if __name__ == "__main__":
    FilterPipeline.load_from_yaml("mia/fpv/filter_pipelines/mia.yaml")