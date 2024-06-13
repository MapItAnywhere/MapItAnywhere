"""

Example usage:
    python3.9 -m mapper.data.debug.calc_stats -d /ocean/projects/cis220039p/shared/map_perception/dataset_v0
"""
import datetime
from datetime import datetime, timezone, timedelta
import time
import argparse
import os
from pathlib import Path
import json

from astral import LocationInfo
from astral.sun import sun
from timezonefinder import TimezoneFinder

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj.transformer import Transformer
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tqdm

from ..fpv import filters
from .. import logger


def is_daytime(timestamp, latitude, longitude):
    # Create a LocationInfo object for the given latitude and longitude
    tz_str = TimezoneFinder().timezone_at(lng=longitude, lat=latitude) 
    location = LocationInfo(name="", region="", timezone=tz_str,
                            latitude=latitude, longitude=longitude)
    
    # Convert the timestamp to a datetime object
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    # We query one day before and one day after to avoid timezone ambiguities
    # Our query timestamp is guaranteed to fall into one of those 3 dates.
    # Astral sometimes returns sunrise or sunsets that are not from the same query date
    # Refer to this https://github.com/sffjunkie/astral/issues/83
    d0 = (dt - timedelta(days=1)).date()
    d1 = dt.date()
    d2 = (dt + timedelta(days=1)).date()

    # Calculate sunrise and sunset times
    times = list()
    for d in [d0, d1, d2]:
        s = sun(location.observer, date=d)
        sunrise = s['sunrise']
        sunset = s['sunset']
        times.append((sunrise, "sunrise"))
        times.append((sunset, 'sunset'))
    
    # Need to sort because there is no particular order 
    # where sunrise is always before sunset or vice versa
    times = sorted(times, key=lambda x: x[0])
    assert times[-1][0] > dt > times[0][0]

    for i in range(1, len(times)):
        if dt < times[i][0]:
            prev_event = times[i-1][1]
            break
    
    return prev_event == "sunrise"

def calculate_occupancy_map(df: pd.DataFrame, bev_meter_coverage=112, meters_per_pixel=112):
    """
    Args:
        bev_meter_coverage: How much did the BEVs in the dataframe cover in meters
        meters_per_pixel: At what resolution should we initialize the occupancy map. 
        This need not be the same resolution as the BEV. That would be unnecessarilly slow but most accurate.
    """
    # convert pandas dataframe to geopandas dataframe
    gdf = gpd.GeoDataFrame(df, 
                           geometry=gpd.points_from_xy(
                               df['computed_geometry.long'], 
                               df['computed_geometry.lat']), 
                            crs=4326)

    utm_crs = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm_crs)
    left = gdf_utm.geometry.x.min() - bev_meter_coverage
    right = gdf_utm.geometry.x.max() + bev_meter_coverage
    bottom = gdf_utm.geometry.y.min() - bev_meter_coverage
    top = gdf_utm.geometry.y.max() + bev_meter_coverage

    width = right - left
    height = top - bottom
    width_pixels = int(width // meters_per_pixel)
    height_pixels = int(height // meters_per_pixel)
    if bev_meter_coverage % meters_per_pixel != 0:
        logger.warn(f"bev_meter_coverage {bev_meter_coverage} is not divisble by meters_per_pixel "
                    f"{meters_per_pixel}. Occupancy may be overestimated.")

    bev_pixels = int(np.ceil(bev_meter_coverage / meters_per_pixel))

    logger.info(f"Initializing {height_pixels}x{width_pixels} occupancy map. Using {bev_pixels}x{bev_pixels} pixels for each BEV.")
    map = np.zeros((height_pixels, width_pixels), dtype=bool)
    
    for row in gdf_utm.itertuples():
        utm_x = row.geometry.x
        utm_y = row.geometry.y
        img_x = int((utm_x - left) // meters_per_pixel)
        img_y = int((utm_y - bottom) // meters_per_pixel)

        bev_pixels_left = bev_pixels // 2
        bev_pixels_right = bev_pixels - bev_pixels_left
        map[img_y - bev_pixels_left: img_y + bev_pixels_right,
            img_x - bev_pixels_left: img_x + bev_pixels_right] = True
    
    return map

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", '-d', type=str, required=True, help="Dataset directory")
    parser.add_argument("--locations", '-l', type=str, default="all",
                        help="Location names in CSV format. Set to 'all' to traverse all locations.")
    parser.add_argument("--plot", action="store_true", help="Store plots per location in PDFs")
    parser.add_argument("--output", "-o", default=None, type=str, help="output json file to store statistics")
    args = parser.parse_args()

    locations = list()
    if args.locations.lower() == "all":
        locations = os.listdir(args.dataset_dir)
        locations = [l for l in locations if os.path.isdir(os.path.join(args.dataset_dir, l))]
    else:
        locations = args.locations.split(",")

    logger.info(f"Parsing {len(locations)} locations..")

    all_locs_stats = dict()

    for location in tqdm.tqdm(locations):
        dataset_dir = Path(args.dataset_dir)
        location_dir = dataset_dir / location
        bev_dir = location_dir / "bev_raw"
        semantic_mask_dir = location_dir / "semantic_masks"
        osm_cache_dir = location_dir / "osm_cache"

        pq_name = 'image_metadata_filtered_processed.parquet'
        df = pd.read_parquet(location_dir / pq_name)

        df = df[df["computed_geometry.lat"].notna()]
        df = df[df["computed_geometry.long"].notna()]

        logger.info(f"Loaded {df.shape[0]} image metadata from {location}")

        # Calc derrivative attributes
        tqdm.tqdm.pandas()

        df["loc_descrip"] = filters.haversine_np(
            lon1=df["geometry.long"], lat1=df["geometry.lat"],
            lon2=df["computed_geometry.long"], lat2=df["computed_geometry.lat"]
        )

        df["angle_descrip"] = filters.angle_dist(
            df["compass_angle"],
            df["computed_compass_angle"]
        )

        # FIXME: Super slow
        # df["is_daytime"] = df.progress_apply(lambda x: is_daytime(x["captured_at"]*1e-3, 
        #                                                  x["computed_geometry.lat"], 
        #                                                  x["computed_geometry.long"]), 
        #                             axis="columns", raw=False, engine="python")

        meters_per_pixel = 7
        map = calculate_occupancy_map(df, bev_meter_coverage=112, 
                                      meters_per_pixel=meters_per_pixel)

        # Calc aggregate stats
        loc_stats = dict()
        loc_stats["num_images"] = len(df)
        loc_stats["area_covered_km2"] = np.sum(map) * meters_per_pixel ** 2 * 1e-6
        loc_stats["camera_types"] = set(df["camera_type"].unique())
        loc_stats["camera_makes"] = set(df["make"].unique())
        loc_stats["camera_model"] = set(df["model"].unique())

        all_locs_stats[location] = loc_stats

        # Plot if requested
        if args.plot:
            with PdfPages(location_dir / "stats.pdf") as pdf:
                plt.figure()
                plt.imshow(map)
                plt.title(f"{location} occupancy map")
                pdf.savefig()
                plt.close()
                for k in ["make", "model", "camera_type", "loc_descrip",
                          "angle_descrip"]:
                    plt.figure()
                    df[k].hist()
                    plt.title(k)
                    plt.xlabel(k)
                    plt.xticks(rotation=90)
                    plt.ylabel("Count")
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()

    # Aggregate all stats
    aggregated_stats = dict()
    for loc, loc_stats in all_locs_stats.items():
        for k,v in loc_stats.items():
            if isinstance(v, float) or isinstance(v, int):
                if k not in aggregated_stats.keys():
                    aggregated_stats[k] = v
                else:
                    aggregated_stats[k] += v
            elif isinstance(v, set):
                if k not in aggregated_stats.keys():
                    aggregated_stats[k] = v
                else:
                    aggregated_stats[k] = aggregated_stats[k].union(v)
                aggregated_stats[f"{k}_count"] = len(aggregated_stats[k])
            else:
                raise Exception(f"{v} is not supported !")

    all_locs_stats["aggregated"] = aggregated_stats

    print(all_locs_stats)

    # Store for json
    for loc, loc_stats in all_locs_stats.items():
        for k,v in loc_stats.items():
            if isinstance(v, set):
                loc_stats[k] = list(v)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_locs_stats, f, indent=2)