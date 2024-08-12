"""python3.9 -m mia.fpv.get_fpv --cfg mia/conf/example.yaml"""

import argparse
import itertools
import traceback
from functools import partial
from typing import Dict
from pathlib import Path
import tracemalloc
import copy
import json

import numpy as np
import asyncio
from tqdm import tqdm
from omegaconf import OmegaConf
import pandas as pd

from .. import logger
from .geo import Projection

from .download import (
    MapillaryDownloader,
    fetch_image_infos,
    fetch_images_pixels,
    get_city_boundary,
    get_tiles_from_boundary,
)
from .prepare import process_sequence, default_cfg
from .filters import in_shape_filter, FilterPipeline

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return json.JSONEncoder.default(self, obj)

def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, cls=JSONEncoder)

def get_token(token: str) -> str:
    if Path(token).is_file():
        logger.info(f"Reading token from file {token}")
        with open(token, 'r') as file:
            token = file.read().strip()

    if not token.startswith("MLY"):
        logger.fatal(f"The token '{token}' is invalid")
        exit(1)
    else:
        logger.info(f"Using token {token}")
    return token

def fetch_city_boundaries(cities: list):
    """
    Args:
        cities: List of dictionaries describing the city/region to fetch in the fpv.yaml format.
    """
    data = []
    pbar = tqdm(cities)
    for loc_info in pbar:
        loc_fmt = loc_info["name"]

        if "state" in loc_info:
            loc_fmt = f"{loc_fmt}, {loc_info['state']}"
        else:
            loc_info["state"] = ""

        if "country" in loc_info:
            loc_fmt = f"{loc_fmt}, {loc_info['country']}"
        else:
            loc_info["country"] = ""
        
        pbar.set_description(f"Getting boundary for {loc_fmt}")
        entry = copy.copy(dict(loc_info))

        get_city_boundary_ = partial(get_city_boundary, loc_info["name"], loc_info["state"], loc_info["country"])
        if "bound_type" not in loc_info:
            assert "sequence_ids" in loc_info
            raise NotImplementedError()
        elif loc_info["bound_type"] == "custom_bbox":
            assert "custom_bbox" in loc_info
            entry["bbox"] = dict(zip(["west", "south", "east", "north"],
                                 [float(x) for x in loc_info["custom_bbox"].split(",")]))
        elif loc_info["bound_type"] == "auto_shape":
            entry["bbox"], entry["shape"] = get_city_boundary_(fetch_shape=True)
        elif loc_info["bound_type"] == "auto_bbox":
            entry["bbox"] = get_city_boundary_(fetch_shape=False)
        elif loc_info["bound_type"] == "custom_size":
            assert "custom_size" in loc_info
            custom_size = loc_info["custom_size"]
            bbox = get_city_boundary_(fetch_shape=False)
            # Calculation below is obviously not very accurate.
            # Good enough for small bounding boxes
            bbox_center = [(bbox['west'] + bbox['east'])/2, (bbox['south'] + bbox['north'])/2]
            bbox['west'] = bbox_center[0] - custom_size / (111.32*np.cos(np.deg2rad(bbox_center[1])))
            bbox['east'] = bbox_center[0] + custom_size / (111.32*np.cos(np.deg2rad(bbox_center[1])))
            bbox['south'] = bbox_center[1] - custom_size / 111.32
            bbox['north'] = bbox_center[1] + custom_size / 111.32
            entry["bbox"] = bbox
            entry["custom_size"] = custom_size
        else:
            raise Exception(f"Unsupported bound_type type '{loc_info['bound_type']}'")

        data.append(entry)
    return data

def geojson_feature_list_to_pandas(feature_list, split_coords=True):
    t = pd.json_normalize(feature_list)
    cols_to_drop = ["type", "geometry.type", "properties.organization_id", "computed_geometry.type"]
    if split_coords:
        t[['geometry.long','geometry.lat']] = pd.DataFrame(t["geometry.coordinates"].tolist(), index=t.index)
        # Computed geometry maybe nan if its not available so we check if the value could be a nan (a float type)
        if "computed_geometry.coordinates" in t.columns:
            t["computed_geometry.long"] = t["computed_geometry.coordinates"].map(lambda x: (x if isinstance(x, float) else x[0]) )
            t["computed_geometry.lat"] = t["computed_geometry.coordinates"].map(lambda x: (x if isinstance(x, float) else x[1]) )

    t.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    t.columns = t.columns.str.removeprefix('properties.')
    t["id"] = t["id"].astype(str)
    return t

def parse_image_points_json_data(rd: dict, combine=True) -> pd.DataFrame:
    """
    Parse the json in to a pandas dataframe
    """
    df_dict = dict()
    for tile, feature_list in tqdm(rd.items(), total=len(rd)):
        if len(feature_list) == 0:
            continue
        df_dict[tile] = geojson_feature_list_to_pandas(feature_list)
    
    if combine:
        logger.info(f"Joining all dataframes into one.")
        return pd.concat(df_dict.values())
    else:
        return df_dict

def log_memory_usage():
    current, peak = tracemalloc.get_traced_memory()
    current_gb = current / 10**9
    peak_gb = peak / 10**9
    logger.info(f"Current memory: {current_gb:.3f} GB; Peak was {peak_gb:.3f} GB")

def main(args, cfgs):
    pipeline = FilterPipeline.load_from_yaml(cfgs.fpv_options.filter_pipeline_cfg)

    # setup the mapillary downloader
    tracemalloc.start()
    token = get_token(args.token)
    downloader = MapillaryDownloader(token)
    loop = asyncio.get_event_loop()

    # setup file structure
    dataset_dir = Path(cfgs.dataset_dir)
    dataset_dir.mkdir(exist_ok=True, parents=True)

    # Fetch the bounds for the cities 
    logger.info(f"Auto fetching boundaries for cities if needed.")
    cities_bounds_info = fetch_city_boundaries(cfgs.cities)

    log_memory_usage()

    # loop through the cities and collect the mapillary data (images, metadata, etc.)
    for city_boundary_info in cities_bounds_info:
        # Clear out dataframes since we may use None checks to see if we need 
        # to load the dataframe for a particular stage
        df = None
        df_meta = None
        df_meta_filtered = None
        df_meta_filtered_processed = None

        logger.info(f"Processing {city_boundary_info['name']}")
        # setup the directories 
        location_name = city_boundary_info['name'].lower().replace(" ", "_")
        location_dir = dataset_dir / location_name
        infos_dir = location_dir / "image_infos_chunked"
        raw_image_dir = location_dir / "images_raw"
        out_image_dir = location_dir / "images"
        for d in (infos_dir, raw_image_dir, out_image_dir, location_dir):
            if not d.exists():
                logger.info(f"{d} does not exist. Creating directory {d}")
                d.mkdir(parents=True, exist_ok=True)
        write_json(location_dir / "boundary_info.json", city_boundary_info)

        # Stage 1: collect the id of the images in the specified bounding box
        if cfgs.fpv_options.stages.get_image_points_from_tiles:
            logger.info(f"[{location_name}] Stage 1 (Downloading image IDs) ------------------")
            tiles = get_tiles_from_boundary(city_boundary_info)
            logger.info(f"[{location_name}] Found {len(tiles)} zoom-14 tiles for this boundary. Starting image point download")
            image_points_response = loop.run_until_complete(
                    downloader.get_tiles_image_points(tiles)
                )
            if image_points_response is None:
                logger.warn(f"[{location_name}] No image points found in boundary. Skipping city")
                continue
            write_json(location_dir / 'images_points_dump.json', image_points_response)

            # parse the data into a geopandas dataframe
            logger.info(f"[{location_name}] Parsing image point json data into dataframe")
            df = parse_image_points_json_data(image_points_response)

            # Filter if needed
            if city_boundary_info["bound_type"] == "auto_shape":
                old_count = df.shape[0]
                df = df[in_shape_filter(df, city_boundary_info["shape"])]
                new_count = df.shape[0]
                logger.info(f"[{location_name}] Keeping {new_count}/{old_count} ({new_count/old_count*100:.2f}%) "
                            "points that are within city boundaries")

            # Image IDs fetched may have duplicates due to an image being on the boundary of two tiles
            # Let us filter those duplicates out if needed
            df_unique = df.drop_duplicates(subset="id")
            if len(df_unique) < len(df):
                logger.warn(f"Image Points fetched from tiles has { len(df)-len(df_unique)} duplicate records. We will drop those.")
            df = df_unique
            
            df.to_parquet(location_dir / 'image_points.parquet')

        # Stage 2: download the metadata
        if cfgs.fpv_options.stages.get_metadata:
            logger.info(f"[{location_name}] Stage 2 (Downloading Metadata) ------------------")
            if df is None:
                pq_name = 'image_points.parquet'
                df = pd.read_parquet(location_dir / pq_name)
                logger.info(f"[{location_name}] Loaded {df.shape[0]} image points from {pq_name}")
                log_memory_usage()
            
            # chunk settings
            chunk_size = cfgs.fpv_options.metadata_download_chunk_size
            num_split = int(np.ceil(df.shape[0] / chunk_size))
            logger.info(f"[{location_name}] Splitting the {df.shape[0]} image points into {num_split} chunks of {chunk_size} image points each.")

            # check if the metadata chunk has already been downloaded
            num_downloaded_chunks = 0
            num_of_chunks_in_dir = len(list(infos_dir.glob("image_metadata_chunk_*.parquet")))
            df_meta_chunks = list()
            df_meta = pd.DataFrame()
            if infos_dir.exists() and num_of_chunks_in_dir > 0:
                logger.info(f"[{location_name}] Found {len(list(infos_dir.glob('image_metadata_chunk_*.parquet')))} existing metadata chunks.")
                downloaded_ids = []
                num_downloaded_data_pts = 0 
                pbar = tqdm(infos_dir.glob("image_metadata_chunk_*.parquet"), total=num_of_chunks_in_dir)
                for chunk_fp in pbar:
                    pbar.set_description(f"Loading {chunk_fp}")
                    chunk_df = pd.read_parquet(chunk_fp)
                    df_meta_chunks.append(chunk_df)
                    num_downloaded_chunks += 1
                    num_downloaded_data_pts += len(chunk_df)
                    log_memory_usage()
                
                num_pts_left = df.shape[0] - num_downloaded_data_pts

                df_meta = pd.concat(df_meta_chunks)
                df_meta_chunks.clear()
                df = df[~df["id"].isin(df_meta["id"])]
                
                # some quick checks to make sure the data is consistent
                left_num_split = int(np.ceil(df.shape[0] / chunk_size))
                # if num_downloaded_chunks != (num_split - left_num_split):
                #     raise ValueError(f"Number of downloaded chunks {num_downloaded_chunks} does not match the number of chunks {num_split - left_num_split}")
                if num_pts_left != len(df):
                    raise ValueError(f"Number of points left {num_pts_left} does not match the number of points in the dataframe {len(df)}")
                
                if num_pts_left > 0:
                    logger.info(f"Restarting metadata download with {num_pts_left} points, {left_num_split} chunks left to download.")

            # download the metadata
            num_split = int(np.ceil(df.shape[0] / chunk_size))
            groups = df.groupby(np.arange(len(df.index)) // chunk_size)
            
            for (frame_num, frame) in groups: 
                frame_num = frame_num + num_downloaded_chunks
                logger.info(f"[{location_name}] Fetching metadata for {frame_num+1}/{num_split} chunk of {frame.shape[0]} image points.")
                image_ids = frame["id"]
                image_infos, num_fail = loop.run_until_complete(
                    fetch_image_infos(image_ids, downloader, infos_dir)
                )
                logger.info("%d failures (%.1f%%).", num_fail, 100 * num_fail / len(image_ids))
                if num_fail == len(image_ids):
                    logger.warn(f"[{location_name}] All images failed to be fetched. Skipping next steps")
                    continue
                new_df_meta = geojson_feature_list_to_pandas(image_infos.values())
                df_meta_chunks.append(new_df_meta)
                new_df_meta.to_parquet(infos_dir / f'image_metadata_chunk_{frame_num}.parquet')
                log_memory_usage()
                     
            # Combine all new chunks into one DF
            df_meta = pd.concat([df_meta] + df_meta_chunks)
            df_meta_chunks.clear()

            # Some standardization of the data
            df_meta["model"] = df_meta["model"].str.lower().str.replace(' ', '').str.replace('_', '')
            df_meta["make"] = df_meta["make"].str.lower().str.replace(' ', '').str.replace('_', '')
            df_meta.to_parquet(location_dir / 'image_metadata.parquet')

        # Stage 3: run filter pipeline
        if cfgs.fpv_options.stages.run_filter:
            logger.info(f"[{location_name}] Stage 3 (Filtering) ------------------")
            
            if df_meta is None:
                pq_name = 'image_metadata.parquet'
                df_meta = pd.read_parquet(location_dir / pq_name)
                logger.info(f"[{location_name}] Loaded {df_meta.shape[0]} image metadata from {pq_name}")
   
            df_meta_filtered = pipeline(df_meta)
            df_meta_filtered.to_parquet(location_dir / f'image_metadata_filtered.parquet')
            if df_meta_filtered.shape[0] == 0:
                logger.warning(f"[{location_name}] No images to download. Moving on to next location.")
                continue
            else:
                logger.info(f"[{location_name}] {df_meta_filtered.shape[0]} images to download.")

        # Stage 4: Download filtered images
        if cfgs.fpv_options.stages.download_images:
            logger.info(f"[{location_name}] Stage 4 (Downloading Images) ------------------")
            if df_meta_filtered is None:
                pq_name = f'image_metadata_filtered.parquet'
                df_meta_filtered = pd.read_parquet(location_dir / pq_name)
                logger.info(f"[{location_name}] Loaded {df_meta_filtered.shape[0]} image metadata from {pq_name}")
                log_memory_usage()
            # filter out the images that have already been downloaded
            downloaded_image_fps = list(raw_image_dir.glob("*.jpg"))
            downloaded_image_ids = [fp.stem for fp in downloaded_image_fps]
            df_to_download = df_meta_filtered[~df_meta_filtered["id"].isin(downloaded_image_ids)]
            logger.info(f"[{location_name}] {len(downloaded_image_ids)} images already downloaded. {df_to_download.shape[0]} images left to download.")
            
            # download the images
            image_urls = list(df_to_download.set_index("id")["thumb_2048_url"].items())
            if len(image_urls) > 0:
                num_fail = loop.run_until_complete(
                    fetch_images_pixels(image_urls, downloader, raw_image_dir)
                )
                logger.info("%d failures (%.1f%%).", num_fail, 100 * num_fail / len(image_urls))

        # Stage 5: process the sequences
        if cfgs.fpv_options.stages.to_process_sequence:
            logger.info(f"[{location_name}] Stage 5 (Sequence Processing) ------------------")
            if df_meta_filtered is None:
                pq_name = f'image_metadata_filtered.parquet'
                df_meta_filtered = pd.read_parquet(location_dir / pq_name)
                logger.info(f"[{location_name}] Loaded {df_meta_filtered.shape[0]} image metadata from {pq_name}")
                log_memory_usage()
            
            # prepare the data for processing
            seq_to_image_ids = df_meta_filtered.groupby('sequence')['id'].agg(list).to_dict()
            lon_center = (city_boundary_info['bbox']['east'] + city_boundary_info['bbox']['west']) / 2
            lat_center = (city_boundary_info['bbox']['north'] + city_boundary_info['bbox']['south']) / 2
            projection = Projection(lat_center, lon_center, max_extent=50e3) # increase to 50km max extent for the projection, otherwise it will throw an error

            df_meta_filtered.index = df_meta_filtered["id"]
            image_infos = df_meta_filtered.to_dict(orient="index")
            process_sequence_args = default_cfg

            log_memory_usage()
            
            # process the sequences
            dump = {}
            logger.info(f"[{location_name}] Processing downloaded sequences..")

            processed_ids = list()

            for seq_id, seq_image_ids in tqdm(seq_to_image_ids.items()):
                try: 
                    d, pi = process_sequence(
                        seq_image_ids,
                        image_infos,
                        projection,
                        process_sequence_args,
                        raw_image_dir,
                        out_image_dir,
                    )
                    if d is None or pi is None:
                        raise Exception("process_sequence returned None")
                    processed_ids.append(pi)
                    # TODO We shouldn't need dumps
                    dump.update(d)

                except Exception as e:
                    logger.error(f"[{location_name}] Failed to process sequence {seq_id} skipping it. Error: {repr(e)}.")
                    logger.error(traceback.format_exc())
            
            write_json(location_dir / "dump.json", dump)

            # TODO: Ideally we want to move the keyframe selection filter to 
            # The filtering pipeline such that we do not download unnecessary
            # Raw Images. But for now, we will filter the dataframe one more time after processing
            processed_ids = list(itertools.chain.from_iterable(processed_ids))
            df_meta_filtered_processed = df_meta_filtered[ df_meta_filtered["id"].isin(processed_ids)]
            logger.info(f"[{location_name}] Final yield after processing is {df_meta_filtered_processed.shape[0]} images.")
            df_meta_filtered_processed.to_parquet(location_dir / f'image_metadata_filtered_processed.parquet')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="mia/conf/example.yaml", help="Path to config yaml file.")
    parser.add_argument("--token", type=str, default='mapillary_key', help="Either a token string or a path to a file containing the token.")
    args = parser.parse_args()

    cfgs = OmegaConf.load(args.cfg)

    main(args, cfgs)
