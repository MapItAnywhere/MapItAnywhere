# Adapted from OrienterNet

import json
from pathlib import Path

import numpy as np
import httpx
import asyncio
from aiolimiter import AsyncLimiter
import tqdm
import requests
import mercantile
import geojson
import turfpy.measurement
from vt2geojson.tools import vt_bytes_to_geojson


from opensfm.pygeometry import Camera, Pose
from opensfm.pymap import Shot

from .. import logger
from .geo import Projection


semaphore = asyncio.Semaphore(100)  # number of parallel threads.
image_filename = "{image_id}.jpg"
info_filename = "{image_id}.json"


class MapillaryDownloader:
    image_fields = (
        "id",
        "height",
        "width",
        "camera_parameters",
        "camera_type",
        "captured_at",
        "compass_angle",
        "geometry",
        "altitude",
        "computed_compass_angle",
        "computed_geometry",
        "computed_altitude",
        "computed_rotation",
        "thumb_2048_url",
        "thumb_original_url",
        "sequence",
        "sfm_cluster",
        "creator",
        "make",
        "model",
        "is_pano",
        "quality_score",
        "exif_orientation"
    )
    image_info_url = (
        "https://graph.mapillary.com/{image_id}?access_token={token}&fields={fields}"
    )
    seq_info_url = "https://graph.mapillary.com/image_ids?access_token={token}&sequence_id={seq_id}"
    tile_info_url = "https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}?access_token={token}"
    max_requests_per_minute = 50_000

    def __init__(self, token: str):
        self.token = token
        self.client = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(retries=20), timeout=600
        )
        self.limiter = AsyncLimiter(self.max_requests_per_minute // 2, time_period=60)

    async def call_api(self, url: str):
        async with self.limiter:
            r = await self.client.get(url)
        if not r.is_success:
            logger.error("Error in API call: %s", r.text)
        return r


    async def get_tile_image_points(self, tile):
        url = self.tile_info_url.format(
            x=tile.x,
            y=tile.y,
            z=tile.z,
            token=self.token
        )
        try :
            r = await self.call_api(url)
            if r.is_success:
                geo_d = vt_bytes_to_geojson(
                    b_content=r._content,
                    x=tile.x,
                    y=tile.y,
                    z=tile.z,
                    layer="image",
                )
                d = geo_d["features"]
                return tile, d
        except Exception as e:
            logger.error(f"{type(e).__name__}: {e}")
        return tile, None

    async def get_tiles_image_points(self, tiles, retries=3):
        tile_to_images = {}
        tasks = [self.get_tile_image_points(t) for t in tiles]
        for i in range(retries):
            failed_tiles = list()
            for task in tqdm.asyncio.tqdm.as_completed(tasks):
                tile, image_ids = await task
                if image_ids is not None:
                    tile_to_images[f"z_{tile.z}_x{tile.x}_y{tile.y}"] = image_ids
                else:
                    logger.error(f"Error when retrieving tile z_{tile.z}_x{tile.x}_y{tile.y}. Image_ids is None. Skipping.")
                    failed_tiles.append(tile)
            if len(failed_tiles) == 0:
                break
            else:
                if i == retries-1:
                    logger.error(f"Failed to retrieve {len(failed_tiles)} tiles in attempt {i}. Maxed out retries. Skipping those tiles.")
                else:
                    logger.error(f"Failed to retrieve {len(failed_tiles)} tiles in attempt {i}. Trying again..")
                    tasks = [self.get_tile_image_points(t) for t in failed_tiles]
        return tile_to_images


    async def get_image_info(self, image_id: int):
        url = self.image_info_url.format(
            image_id=image_id,
            token=self.token,
            fields=",".join(self.image_fields),
        )
        r = await self.call_api(url)
        if r.is_success:
            return json.loads(r.text)

    async def get_sequence_info(self, seq_id: str):
        url = self.seq_info_url.format(seq_id=seq_id, token=self.token)
        r = await self.call_api(url)
        if r.is_success:
            return json.loads(r.text)

    async def download_image_pixels(self, url: str, path: Path):
        r = await self.call_api(url)
        if r.is_success:
            with open(path, "wb") as fid:
                fid.write(r.content)
        return r.is_success

    async def get_image_info_cached(self, image_id: int, path: Path):
        if path.exists():
            info = json.loads(path.read_text())
        else:
            info = await self.get_image_info(image_id)
            path.write_text(json.dumps(info))
        return info

    async def download_image_pixels_cached(self, url: str, path: Path):
        if path.exists():
            return True
        else:
            return await self.download_image_pixels(url, path)


async def fetch_images_in_sequence(i, downloader):
    async with semaphore:
        info = await downloader.get_sequence_info(i)
    image_ids = [int(d["id"]) for d in info["data"]]
    return i, image_ids


async def fetch_images_in_sequences(sequence_ids, downloader):
    seq_to_images_ids = {}
    tasks = [fetch_images_in_sequence(i, downloader) for i in sequence_ids]
    for task in tqdm.asyncio.tqdm.as_completed(tasks):
        i, image_ids = await task
        seq_to_images_ids[i] = image_ids
    return seq_to_images_ids


async def fetch_image_info(i, downloader, dir_):
    async with semaphore:
        path = dir_ / info_filename.format(image_id=i)
        # info = await downloader.get_image_info_cached(i, path)
        info = await downloader.get_image_info(i) # FIXME: temporarily disable caching, takes too long to reads many (>1mil) files
    return i, info


async def fetch_image_infos(image_ids, downloader, dir_):
    infos = {}
    num_fail = 0
    tasks = [fetch_image_info(i, downloader, dir_) for i in image_ids]
    for task in tqdm.asyncio.tqdm.as_completed(tasks):
        i, info = await task
        if info is None:
            num_fail += 1
        else:
            infos[i] = info
    return infos, num_fail


async def fetch_image_pixels(i, url, downloader, dir_, overwrite=False):
    async with semaphore:
        path = dir_ / image_filename.format(image_id=i)
        if overwrite:
            path.unlink(missing_ok=True)
        success = await downloader.download_image_pixels_cached(url, path)
    return i, success


async def fetch_images_pixels(image_urls, downloader, dir_):
    num_fail = 0
    tasks = [fetch_image_pixels(*id_url, downloader, dir_) for id_url in image_urls]
    for task in tqdm.asyncio.tqdm.as_completed(tasks):
        i, success = await task
        num_fail += not success
    return num_fail


def opensfm_camera_from_info(info: dict) -> Camera:
    cam_type = info["camera_type"]
    if cam_type == "perspective":
        camera = Camera.create_perspective(*info["camera_parameters"])
    elif cam_type == "fisheye":
        camera = Camera.create_fisheye(*info["camera_parameters"])
    elif Camera.is_panorama(cam_type):
        camera = Camera.create_spherical()
    else:
        raise ValueError(cam_type)
    camera.width = info["width"]
    camera.height = info["height"]
    camera.id = info["id"]
    return camera


def opensfm_shot_from_info(info: dict, projection: Projection) -> Shot:
    latlong = info["computed_geometry.coordinates"][::-1]
    alt = info["computed_altitude"]
    xyz = projection.project(np.array([*latlong, alt]), return_z=True)
    c_rotvec_w = np.array(info["computed_rotation"])
    pose = Pose()
    pose.set_from_cam_to_world(-c_rotvec_w, xyz)
    camera = opensfm_camera_from_info(info)
    return latlong, Shot(info["id"], camera, pose)


def get_city_boundary(city, state=None, country=None, fetch_shape=False):
    # Use Nominatim API to get the boundary of the city
    base_url = "https://nominatim.openstreetmap.org/search"
    params = {
        'city': city,
        'state': state,
        'country': country,
        'format': 'json',
        'limit': 1,
        'polygon_geojson': 1 if fetch_shape else 0
    }

    # Without a user-agent we may get blocked. This is an arbitrary user-agent and can be changed
    # Rotating between user-agents may circumvent blocks but may not be fair
    headers = {
        'User-Agent': f'mapperceptionnet_{city}_{state}'
    }
    response = requests.get(base_url, params=params, headers=headers)

    if response.status_code != 200:
        logger.error(f"Nominatim error when fetching boundary data for {city}, {state}.\n"
                     f"Status code: {response.status_code}. Content: {response.content}")
        return None
    
    data = response.json()

    if data is None:
        logger.warn(f"No data returned by Nominatim for {city}, {state}")
        return None
    
    # Extract bbox data from the API response
    bbox_data = data[0]['boundingbox']
    bbox = {
        'west': float(bbox_data[2]),
        'south': float(bbox_data[0]),
        'east': float(bbox_data[3]),
        'north': float(bbox_data[1])
    }

    if fetch_shape:
        # Extract GeoJSON boundary data from the API response
        boundary_geojson = data[0]['geojson']
        boundary_geojson = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
            "properties": {},
            "geometry": boundary_geojson}]
        }
        return bbox, boundary_geojson
    else:
        return bbox


def get_tiles_from_boundary(boundary_info, zoom=14):
    if boundary_info["bound_type"] == "auto_shape":
        # TODO: Instead of tiles from the big bbox, return tiles that hug the shape
        geojson_shape = boundary_info["shape"]
        
        # FIXME What to do when boundary is defined by multiple polygons!!
        # Visualization tool https://geojson.tools/
        coords = geojson_shape["features"][0]["geometry"]["coordinates"]
        try:
            polygon = geojson.Polygon(coords)
            coordinates = turfpy.measurement.bbox(polygon)
        except:
            logger.warn(f"Boundary is defined by {len(coords)} polygons. Choosing first polygon blindly")
            polygon = geojson.Polygon(coords[0])
            coordinates = turfpy.measurement.bbox(polygon)
        
        coordinates = dict(zip(["west", "south", "east", "north"], coordinates))
    else:
        coordinates = boundary_info["bbox"]
        
    tiles = list(
            mercantile.tiles(
                **coordinates,
                zooms=zoom,
            )
        )

    return tiles