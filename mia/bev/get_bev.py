"""Script to get BEV images from a dataset of locations.

Example usage:
    python3.9 -m mia.bev.get_bev
"""

import argparse
import multiprocessing as mp
from pathlib import Path
import io
import os
import requests
import contextlib
import traceback
import colour

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import geopandas as gpd
import torch.nn as nn
import torch
from tqdm import tqdm
from filelock import FileLock
from math import sqrt, ceil
import svgwrite
import cairosvg
from PIL import Image
from xml.etree import ElementTree as ET
from pyproj.transformer import Transformer
from shapely.geometry import box
from omegaconf import OmegaConf
import urllib3

from map_machine.map_configuration import MapConfiguration
from map_machine.scheme import Scheme
from map_machine.geometry.boundary_box import BoundaryBox
from map_machine.osm.osm_getter import NetworkError
from map_machine.osm.osm_reader import OSMData
from map_machine.geometry.flinger import MercatorFlinger
from map_machine.pictogram.icon import ShapeExtractor
from map_machine.workspace import workspace
from map_machine.mapper import Map
from map_machine.constructor import Constructor

from .. import logger
from .image import center_crop_to_size, center_pad

# MUST match colors from map rendering style
COLORS = {
    "road": "#000",
    "crossing": "#F00",
    "explicit_pedestrian": "#FF0",
    "park": "#0F0",
    "building": "#F0F",
    "water": "#00F",
    "terrain": "#0FF",
    "parking": "#AAA",
    "train": "#555"
}

# While the color mapping above must match what is in the 
# rendering style, the pretty colors below are just for visualization
# purposes and can easily be changed below without worrying.
# Colors set to None will not be rendered in rendered masks
PRETTY_COLORS = {
    "road": "#444",
    "crossing": "#F4A261",
    "explicit_pedestrian": "#E9C46A",
    "park": None,
    "building": "#E76F51",
    "water": None,
    "terrain": "#2A9D8F",
    "parking": "#CCC",
    "train": None
}

# Better order for visualization
VIS_ORDER = ["terrain", "water", "park", "parking", "train", 
             "road", "explicit_pedestrian", "crossing", "building"]

def checkColor(code):

    def check_ele(ele):
        isColor = False
        if "stroke" in ele.attribs:
            if ele.attribs["stroke"] != "none":
                color = colour.Color(ele.attribs["stroke"])
                isColor |= color == colour.Color(code)
        
        if "fill" in ele.attribs:
            if ele.attribs["fill"] != "none":
                color = colour.Color(ele.attribs["fill"])
                isColor |= color == colour.Color(code)

        return isColor
    
    return check_ele

def hex2rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    if len(hex_str) == 3:
        hex_str = "".join([hex_str[i//2] for i in range(6)])
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

def mask2rgb(mask, pretty=True):
    H,W,N = mask.shape
    rgb = np.ones((H,W,3), dtype=np.uint8)*255
    cmap = PRETTY_COLORS if pretty else COLORS
    key2mask_i = dict(zip(cmap.keys(), range(N)))
    for k in VIS_ORDER:
        if cmap[k]:
            rgb[mask[:,:, key2mask_i[k]]>0.5] = (np.array(hex2rgb(cmap[k])))
    
    return rgb

def draw_bev(bbox: BoundaryBox, osm_data: OSMData,
             configuration: MapConfiguration, meters_per_pixel: float, heading: float):
    """Rasterize OSM data as a BEV image"""
    lat = bbox.center()[0]
    # Equation rearranged from https://wiki.openstreetmap.org/wiki/Zoom_levels
    # To get zoom level given meters_per_pixel
    z = np.log2(np.abs(osm_data.equator_length*np.cos(np.deg2rad(lat))/meters_per_pixel/256))
    flinger = MercatorFlinger(bbox, z, osm_data.equator_length)

    size = flinger.size
    svg: svgwrite.Drawing = svgwrite.Drawing(None, size) # None since we are not saving an svg file
    
    icon_extractor: ShapeExtractor = ShapeExtractor(
        workspace.ICONS_PATH, workspace.ICONS_CONFIG_PATH
    )
    constructor: Constructor = Constructor(
        osm_data=osm_data,
        flinger=flinger,
        extractor=icon_extractor,
        configuration=configuration,
    )
    constructor.construct()
    map_: Map = Map(flinger=flinger, svg=svg, configuration=configuration)
    try:
        imgs = []

        map_.draw(constructor)

        # svg.defs.add(svgwrite.container.Style(f"transform: rotate({str(heading)}deg)"))
        for ele in svg.elements:
            ele.rotate(360 - heading, (size[0]/2, size[1]/2))

        for k, v in COLORS.items():
            svg_new = svg.copy()
            svg_new.elements = list(filter(checkColor(v), svg_new.elements))
        
            png_byte_string = cairosvg.svg2png(bytestring=svg_new.tostring(), 
                                            output_width=size[0], 
                                            output_height=size[1]) # convert svg to png
            img = Image.open(io.BytesIO(png_byte_string))

            imgs.append(img)

    except Exception as e:
        # Prepare the stack trace
        stack_trace = traceback.format_exc()
        logger.error(f"Failed to render BEV for bbox {bbox.get_format()}. Exception: {repr(e)}. Skipping.. Stack trace: {stack_trace}")
        return None, None

    return imgs, svg


def process_img(img, num_pixels, heading=None):
    """Rotate + Crop to correct for heading and ensure correct dimensions"""

    img = center_pad(img, num_pixels, num_pixels)
    s = min(img.size)
    squared_img = center_crop_to_size(img, s, s) # Ensure it is square before rotating (Perhaps not needed)
    if heading:
        squared_img = squared_img.rotate(heading, expand=False, resample=Image.Resampling.BILINEAR)
    center_cropped_bev_img = center_crop_to_size(squared_img, num_pixels, num_pixels)
    # robot_cropped_bev_img = center_cropped_bev_img.crop((0, 0, num_pixels, num_pixels/2))  # left, upper, right, lower
    return center_cropped_bev_img


def get_satellite_from_bbox(bbox, output_fp, num_pixels, heading):
    # TODO: This method does not always produce a full satellite image.
    # We need something more consistent like mapbox but free.

    region = ee.Geometry.Rectangle(bbox, proj="EPSG:4326", geodesic=False)
    # Load a satellite image collection, filter it by date and region, then select the first image
    image = ee.ImageCollection('USDA/NAIP/DOQQ') \
        .filterBounds(region) \
        .filterDate('2022-01-01', '2022-12-31') \
        .sort('CLOUDY_PIXEL_PERCENTAGE') \
        .first().select(['R', 'G', 'B'])

    # Reproject the image to a common projection (e.g., EPSG:4326)
    image = image.reproject(crs='EPSG:4326', scale=0.5)

    # Get the image URL
    url = image.getThumbURL({'min': 0, 'max': 255, 'region': region.getInfo()['coordinates']})

    # Download the image to your desktop
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content))
    robot_cropped_bev_img = process_img(img, num_pixels, heading)
    robot_cropped_bev_img.save(output_fp)


def get_data(address: str, parameters: dict[str, str]) -> bytes:
    """
    Construct Internet page URL and get its descriptor.

    :param address: URL without parameters
    :param parameters: URL parameters
    :return: connection descriptor
    """
    for _ in range(50):
        http = urllib3.PoolManager()

        urllib3.disable_warnings()

        try:
            result = http.request("GET", address, fields=parameters)
        except urllib3.exceptions.MaxRetryError:
            continue
        
        if result.status == 200:
            break
    else:
        print(result.data)
        raise NetworkError(f"Cannot download data: {result.status} {result.reason}")
    
    http.clear()
    return result.data


def get_osm_data(bbox: BoundaryBox, osm_output_fp: Path,
                 overwrite=False, use_lock=False) -> OSMData:
    """
    Get OSM data within bounding box from usingoverpass APIs and 
    write data to osm_output_fp.
    """

    OVERPASS_ENDPOINTS = [
        "http://overpass-api.de/api/map",
        "http://overpass.kumi.systems/api/map",
        "http://maps.mail.ru/osm/tools/overpass/api/map"
    ]

    RETRIES = 10
    osm_data = None
    overpass_endpoints_i = 0

    for retry in range(RETRIES):
        try:
            # fetch or load from cache
            # A lock is needed if we are using multiple processes without store_osm_per_id
            # Since multiple workers may share the same cached OSM file.
            # Note: Can optimize locking further by implementing a readers-writer lock scheme
            if use_lock:
                lock_fp = osm_output_fp.parent.parent / (osm_output_fp.parent.name + "_tmp_locks") / (osm_output_fp.name + ".lock")
                lock = FileLock(lock_fp)
            else:
                lock = contextlib.nullcontext()
            
            with lock: 
                if not overwrite and osm_output_fp.is_file():
                    with osm_output_fp.open(encoding="utf-8") as output_file:
                        xml_str = output_file.read()
                else:
                    content: bytes = get_data(
                        address=OVERPASS_ENDPOINTS[overpass_endpoints_i],
                        parameters={"bbox": bbox.get_format()}
                    )

                    xml_str = content.decode("utf-8")

                    if not content.startswith(b"<?xml"):
                        raise Exception(f"Invalid content received: '{xml_str}'")

                    with osm_output_fp.open("bw+") as output_file:
                        output_file.write(content)

            # parse OSM xml string
            tree = ET.fromstring(xml_str)
            osm_data = OSMData()
            osm_data.parse_osm(tree, parse_nodes=True, 
                               parse_relations=True, parse_ways=True)
            break
            
        except Exception as e:
            msg = f"Error: Unable to fetch OSM data for bbox {bbox.get_format()} "\
                  f"for file {osm_output_fp} after {retry+1}/{RETRIES} attempts. Exception: {repr(e)}."
            
            if retry < RETRIES-1:
                overpass_endpoints_i = (overpass_endpoints_i+1) % len(OVERPASS_ENDPOINTS)
                logger.error(f"{msg}. Retrying with {OVERPASS_ENDPOINTS[overpass_endpoints_i]} endpoint..")
                continue
            else:
                logger.error(f"{msg}. Skipping..")   
                break

    return osm_data, retry+1

def get_bev_from_bbox(
    bbox: BoundaryBox, 

    num_pixels: int,
    meters_per_pixel: float,
    configuration: MapConfiguration,

    osm_output_fp: Path,
    bev_output_fp: Path,
    mask_output_fp: Path,
    rendered_mask_output_fp: Path,

    osm_data: OSMData=None,
    heading: float=0,
    final_downsample: int=1,

    download_osm_only: bool=False,
    use_osm_cache_lock: bool=False,
) -> None:
    """Get BEV image from a boundary box. Optionally rotate, crop and save the extracted semantic mask."""

    if osm_data is None: 
        if osm_output_fp.is_file():
            # Load from cache
            try:
                osm_data = OSMData()
                with osm_output_fp.open(encoding="utf-8") as output_file:
                    xml_str = output_file.read()
                    tree = ET.fromstring(xml_str)
                    osm_data.parse_osm(tree, parse_nodes=True, 
                                    parse_relations=True, parse_ways=True)
            except Exception as e:
                osm_data, _ = get_osm_data(bbox, osm_output_fp, use_lock=use_osm_cache_lock)
        else:
        # No local osm planet dump file. Need to download or read from cache
            osm_data, _ = get_osm_data(bbox, osm_output_fp, use_lock=use_osm_cache_lock)
        
        if osm_data is None:
            return

    if download_osm_only:
        return
    
    imgs, svg = draw_bev(bbox, osm_data, configuration, meters_per_pixel, heading)
    if imgs is None:
        return
    
    if bev_output_fp:
        svg.saveas(bev_output_fp)

    cropped_imgs = []
    for img in imgs:
        # Set heading to None because we already rotated in draw_bev
        cropped_imgs.append(process_img(img, num_pixels, heading=None))

    masks = []
    for img in cropped_imgs:
        arr = np.array(img)
        masks.append(arr[..., -1] != 0)
    
    extracted_mask = np.stack(masks, axis=0)
    extracted_mask[2][extracted_mask[0]] = 0

    if final_downsample > 1:
        max_pool_layer = nn.MaxPool2d(kernel_size=final_downsample, stride=final_downsample)
        # Apply max pooling
        mask_tensor = torch.tensor(extracted_mask, dtype=torch.float32).unsqueeze(0)
        max_pool_tensor = max_pool_layer(mask_tensor)
        # Remove the batch dimension and permute back to original dimension order, then convert to numpy
        multilabel_mask_downsampled = max_pool_tensor.squeeze(0).permute(1, 2, 0).numpy()
    else:
        multilabel_mask_downsampled = extracted_mask.transpose(1, 2, 0)


    # Save npz files for semantic masks
    if mask_output_fp:
        np.savez_compressed(mask_output_fp, multilabel_mask_downsampled)

    # Save rendered BEV map if we want for visualization
    if rendered_mask_output_fp:
        rgb = mask2rgb(multilabel_mask_downsampled)
        plt.imsave(rendered_mask_output_fp.with_suffix('.png'), rgb)


def get_bev_from_bbox_worker_init(osm_cache_dir, bev_dir, semantic_mask_dir, rendered_mask_dir,
                                  scheme_path, map_length, meters_per_pixel,
                                  osm_data, redownload, download_osm_only, store_osm_per_id,
                                  use_osm_cache_lock, final_downsample):
    global worker_kwargs
    worker_kwargs=locals()
    # MapConfiguration is not picklable so we have to initialize it for each worker
    scheme = Scheme.from_file(Path(scheme_path))
    configuration = MapConfiguration(scheme)
    configuration.show_credit = False
    worker_kwargs["configuration"] = configuration
    logger.info(f"Worker {os.getpid()} started.")
    

def get_bev_from_bbox_worker(job_dict):
    id = job_dict['id']
    bbox = job_dict['bbox_formatted']
    bbox = BoundaryBox.from_text(bbox)
    heading = job_dict['computed_compass_angle']

    # Setting a path to None disables storing that file
    bev_fp = worker_kwargs["bev_dir"]
    if bev_fp:
        bev_fp = bev_fp / f"{id}.svg"
    
    semantic_mask_fp = worker_kwargs["semantic_mask_dir"]
    if semantic_mask_fp:
        semantic_mask_fp = semantic_mask_fp / f"{id}.npz"
    
    rendered_mask_fp = worker_kwargs["rendered_mask_dir"]
    if rendered_mask_fp:
        rendered_mask_fp = rendered_mask_fp / f"{id}.png"

    if worker_kwargs["store_osm_per_id"]:
        osm_output_fp = worker_kwargs["osm_cache_dir"] / f"{id}.osm"
    else:
        osm_output_fp = worker_kwargs["osm_cache_dir"] / f"{bbox.get_format()}.osm"

    
    if (    (bev_fp           is None or bev_fp.exists()          ) # Bev exists or we don't want to save it
        and (semantic_mask_fp is None or semantic_mask_fp.exists()) # ...
        and (rendered_mask_fp is None or rendered_mask_fp.exists()) # ...
        and not worker_kwargs["redownload"]):
        return
    
    get_bev_from_bbox(bbox=bbox,
                      num_pixels=worker_kwargs["map_length"], 
                      meters_per_pixel=worker_kwargs["meters_per_pixel"],
                      configuration=worker_kwargs["configuration"], 
                      osm_output_fp=osm_output_fp, 
                      bev_output_fp=bev_fp, 
                      mask_output_fp=semantic_mask_fp, 
                      rendered_mask_output_fp=rendered_mask_fp,
                      osm_data=worker_kwargs["osm_data"],
                      heading=heading,
                      final_downsample=worker_kwargs["final_downsample"],
                      download_osm_only=worker_kwargs["download_osm_only"],
                      use_osm_cache_lock=worker_kwargs["use_osm_cache_lock"])

def main(dataset_dir, locations, args):
    # setup directory paths
    dataset_dir = Path(dataset_dir)

    for loc in locations:
        loc_name = loc["name"].lower().replace(" ", "_")
        location_dir = dataset_dir / loc_name
        osm_cache_dir = location_dir / "osm_cache"
        bev_dir = location_dir / "bev_raw" if args.store_all_steps else None
        semantic_mask_dir = location_dir / "semantic_masks" 
        rendered_mask_dir = location_dir / "rendered_semantic_masks" if args.store_all_steps else None

        for d in [location_dir, osm_cache_dir, bev_dir, semantic_mask_dir, rendered_mask_dir]:
            if d:
                d.mkdir(parents=True, exist_ok=True)
        
        # read the parquet file
        parquet_fp = location_dir / f"image_metadata_filtered_processed.parquet"
        logger.info(f"Reading parquet file from {parquet_fp}.")
        df = pd.read_parquet(parquet_fp)

        if args.n_samples > 0:# If -1, use all samples
            logger.info(f"Sampling {args.n_samples} rows.")
            df = df.sample(args.n_samples, replace=False, random_state=1)

        df.reset_index(drop=True, inplace=True)
        logger.info(f"Read {len(df)} rows from the parquet file.")
        
        # convert pandas dataframe to geopandas dataframe
        gdf = gpd.GeoDataFrame(df, 
                            geometry=gpd.points_from_xy(
                                df['computed_geometry.long'], 
                                df['computed_geometry.lat']), 
                                crs=4326)
            
        # convert the geopandas dataframe to UTM
        utm_crs = gdf.estimate_utm_crs()
        gdf_utm = gdf.to_crs(utm_crs)
        transformer = Transformer.from_crs(utm_crs, 4326)
        logger.info(f"UTM zone for {loc_name} is {utm_crs.to_epsg()}.")

        # load OSM data, if available
        padding = args.padding
        # calculate the required distance from the center to the edge of the image 
        # so that the image will not be out of bounds when we rotate it
        map_length = args.map_length
        map_length = ceil(sqrt(map_length**2 + map_length**2))
        distance = map_length * args.meters_per_pixel / 2
        logger.info(f"Each image will be {map_length:.2f} x {map_length:.2f} pixels. The distance from the center to the edge is {distance:.2f} meters.")

        osm_data = None
        if args.osm_fp:
            logger.info(f"Loading OSM data from {args.osm_fp}.")
            osm_fp = Path(args.osm_fp)
            osm_data = OSMData()
            if osm_fp.suffix == '.osm':
                osm_data.parse_osm_file(osm_fp)
            elif osm_fp.suffix == '.json':
                osm_data.parse_overpass(osm_fp)
            else:
                raise ValueError(f"OSM file format {osm_fp.suffix} is not supported.")
            # make sure that the loaded osm data at least covers some points in the dataframe
            bbox = osm_data.boundary_box
            shapely_bbox = box(bbox.left, bbox.bottom, bbox.right, bbox.top)
            logger.warning(f"Clipping the geopandas dataframe to the OSM boundary box. May result in loss of points.")
            gdf = gpd.clip(gdf, shapely_bbox)
            if gdf.empty:
                raise ValueError("Clipped geopandas dataframe is empty. Exiting.")
            logger.info(f"Clipped geopandas dataframe is left with {len(gdf)} points.")

        elif args.one_big_osm:
            osm_fp = location_dir / "one_big_map.osm"
            min_long = gdf_utm.geometry.x.min() - distance - padding
            max_long = gdf_utm.geometry.x.max() + distance + padding
            min_lat = gdf_utm.geometry.y.min() - distance - padding
            max_lat = gdf_utm.geometry.y.max() + distance + padding
            padding = 0
            big_bbox = transformer.transform_bounds(left=min_long, bottom=min_lat, right=max_long, top=max_lat)
            # TODO: Check why transformer is flipping lat long
            big_bbox = (big_bbox[1], big_bbox[0], big_bbox[3], big_bbox[2])
            big_bbox_fmt = ",".join([str(x) for x in big_bbox])
            logger.info(f"Fetching one big osm file using coordinates {big_bbox_fmt}.")
            big_bbox = BoundaryBox.from_text(big_bbox_fmt)
            osm_data, retries = get_osm_data(big_bbox, osm_fp, overwrite=args.redownload)

        # create bounding boxes for each point
        gdf_utm['bounding_box_utm_p1'] = gdf_utm.apply(lambda row: (
            row.geometry.x - distance - padding,
            row.geometry.y - distance - padding,
        ), axis=1)

        gdf_utm['bounding_box_utm_p2'] = gdf_utm.apply(lambda row: (
            row.geometry.x + distance + padding,
            row.geometry.y + distance + padding,
        ), axis=1)

        # convert the bounding box back to lat, long
        gdf_utm['bounding_box_lat_long_p1'] = gdf_utm.apply(lambda row: transformer.transform(*row['bounding_box_utm_p1']), axis=1)
        gdf_utm['bounding_box_lat_long_p2'] = gdf_utm.apply(lambda row: transformer.transform(*row['bounding_box_utm_p2']), axis=1)
        gdf_utm['bbox_min_lat'] = gdf_utm['bounding_box_lat_long_p1'].apply(lambda x: x[0])
        gdf_utm['bbox_min_long'] = gdf_utm['bounding_box_lat_long_p1'].apply(lambda x: x[1])
        gdf_utm['bbox_max_lat'] = gdf_utm['bounding_box_lat_long_p2'].apply(lambda x: x[0])
        gdf_utm['bbox_max_long'] = gdf_utm['bounding_box_lat_long_p2'].apply(lambda x: x[1])
        gdf_utm['bbox_formatted'] = gdf_utm.apply(lambda row: f"{row['bbox_min_long']:.20f},{row['bbox_min_lat']:.20f},{row['bbox_max_long']:.20f},{row['bbox_max_lat']:.20f}", axis=1)

        # iterate over the dataframe and get BEV images
        jobs = gdf_utm[['id', 'bbox_formatted', 'computed_compass_angle']] # only need the id and bbox_formatted columns for the jobs
        jobs = jobs.to_dict(orient='records').copy()

        use_osm_cache_lock = args.n_workers > 0 and not args.store_osm_per_id
        if use_osm_cache_lock:
            logger.info("Using osm cache locks to prevent race conditions since number of workers > 0 and store_osm_per_id is false")
        
        init_args = [osm_cache_dir, bev_dir, semantic_mask_dir, rendered_mask_dir,
                    args.map_machine_scheme, 
                    args.map_length, args.meters_per_pixel, 
                    osm_data, args.redownload, args.download_osm_only, 
                    args.store_osm_per_id, use_osm_cache_lock, args.final_downsample]
        
        if args.n_workers > 0:
            logger.info(f"Launching {args.n_workers} workers to fetch BEVs for {len(jobs)} bounding boxes.")
            with mp.Pool(args.n_workers, 
                        initializer=get_bev_from_bbox_worker_init, 
                        initargs=init_args) as pool:
                for _ in tqdm(pool.imap_unordered(get_bev_from_bbox_worker, jobs, chunksize=16),
                            total=len(jobs), desc="Getting BEV images"):
                    pass
        else:
            get_bev_from_bbox_worker_init(*init_args)
            pbar = tqdm(jobs, desc="Getting BEV images")
            for job_dict in pbar:
                get_bev_from_bbox_worker(job_dict)
        
        # Download sattelite images if needed
        if args.store_sat:
            logger.info("Downloading sattelite images.")
            sat_dir = location_dir / "sattelite"
            sat_dir.mkdir(parents=True, exist_ok=True)
            pbar = tqdm(jobs, desc="Getting Sattelite images")
            for job_dict in pbar:
                id = job_dict['id']
                sat_fp = sat_dir / f"{id}.png"
                if sat_fp.exists() and not args.redownload:
                    continue
                bbox = [float(x) for x in job_dict['bbox_formatted'].split(",")]
                try:
                    get_satellite_from_bbox(bbox, sat_fp, heading=job_dict['computed_compass_angle'], num_pixels=args.map_length)
                except Exception as e:
                    logger.error(f"Failed to get sattelite image for bbox {job_dict['bbox_formatted']}. Exception {repr(e)}")

        # TODO: Post BEV retireval filtering
        # df.to_parquet(location_dir / "image_metadata_bev_processed.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get BEV images from a dataset of locations using MapMachine.")
    parser.add_argument("--cfg", type=str, default="mia/conf/example.yaml", help="Path to config yaml file.")
    args = parser.parse_args()

    cfgs = OmegaConf.load(args.cfg)

    if cfgs.bev_options.store_sat:
        if cfgs.bev_options.n_workers > 0:
            logger.fatal("Satellite download is not multiprocessed yet !!")
        import ee
        ee.Initialize()

    logger.info("="*80)
    logger.info("Running get_bev.py")
    logger.info("Arguments:")
    for arg in vars(args):
        logger.info(f"- {arg}: {getattr(args, arg)}")
    logger.info("="*80)
    main(cfgs.dataset_dir, cfgs.cities, cfgs.bev_options)