from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path
import numpy as np 
from collections import deque
import argparse
import cv2

def get_raycast_building_mask(building_grid):
    laser_range = 200
    num_laser = 100
    robot_pos = (building_grid.shape[0] // 2-1,  building_grid.shape[1] // 2 - 1)
    unoccupied_pos = np.stack(np.where(building_grid != 1), axis=1)

    if len(unoccupied_pos) == 0:
        return None

    l2_dist = unoccupied_pos - [robot_pos[0], robot_pos[1]]
    closest = ((l2_dist ** 2).sum(1)**0.5).argmin()

    robot_pos = (unoccupied_pos[closest][0], unoccupied_pos[closest][1])

    free_points, hit_points, actual_hit_points = get_free_points_in_front(building_grid, robot_pos, laser_range=laser_range, num_laser=num_laser)
    free_points[:, 0][free_points[:, 0] >= building_grid.shape[0]] = building_grid.shape[0] - 1
    free_points[:, 1][free_points[:, 1] >= building_grid.shape[1]] = building_grid.shape[1] - 1
    free_points[:, 0][free_points[:, 0] < 0] = 0
    free_points[:, 1][free_points[:, 1] < 0] = 0

    hit_points[:, 0][hit_points[:, 0] >= building_grid.shape[0]] = building_grid.shape[0] - 1
    hit_points[:, 1][hit_points[:, 1] >= building_grid.shape[1]] = building_grid.shape[1] - 1
    hit_points[:, 0][hit_points[:, 0] < 0] = 0
    hit_points[:, 1][hit_points[:, 1] < 0] = 0

    if len(free_points) > 0:

        # Get vis mask by flood filling free space boundary
        inited_flood_grid = init_flood_fill(robot_pos, hit_points, building_grid.shape)
        inited_flood_grid = (inited_flood_grid * 255).astype(np.uint8).copy()

        # pick a seed point from free points, that is not 0 in inited_flood_grid. We want it to be unknown
        np.random.shuffle(free_points)

        for i in range(len(free_points)):
            seed_point = free_points[i]
            if inited_flood_grid[seed_point[0], seed_point[1]] != 0:
                break  # Found a valid seed point, exit the loop
        else:
            print('Unable to find a valid seed point')
            return None
        
        num_filled, flooded_image, mask, bounding_box = cv2.floodFill(inited_flood_grid.copy(), None, seedPoint=(seed_point[1], seed_point[0]), newVal=0)
        # name = names[batch_ind][-1]
        return flooded_image
    else:
        print("No free points")
        return None

def flood_fill_simple(center_point, occupancy_map):
    """
    center_point: starting point (x,y) of fill
    occupancy_map: occupancy map generated from Bresenham ray-tracing
    """
    # Fill empty areas with queue method
    occupancy_map = np.copy(occupancy_map)
    sx, sy = occupancy_map.shape
    fringe = deque()
    fringe.appendleft(center_point)
    while fringe:
        
        n = fringe.pop()
        nx, ny = n
        unknown_val = 0.5
        # West
        if nx > 0:
            if occupancy_map[nx - 1, ny] == unknown_val:
                occupancy_map[nx - 1, ny] = 0
                fringe.appendleft((nx - 1, ny))
        # East
        if nx < sx - 1:
            if occupancy_map[nx + 1, ny] == unknown_val:
                occupancy_map[nx + 1, ny] = 0
                fringe.appendleft((nx + 1, ny))
        # North
        if ny > 0:
            if occupancy_map[nx, ny - 1] == unknown_val:
                occupancy_map[nx, ny - 1] = 0
                fringe.appendleft((nx, ny - 1))
        # South
        if ny < sy - 1:
            if occupancy_map[nx, ny + 1] == unknown_val:
                occupancy_map[nx, ny + 1] = 0
                fringe.appendleft((nx, ny + 1))
    return occupancy_map

def init_flood_fill(robot_pos, obstacle_points, occ_grid_shape):
    """
    center_point: center point
    obstacle_points: detected obstacles points (x,y)
    xy_points: (x,y) point pairs
    """
    center_x, center_y = robot_pos
    prev_ix, prev_iy = center_x, center_y
    occupancy_map = (np.ones(occ_grid_shape)) * 0.5
    # append first obstacle point to last
    obstacle_points = np.vstack((obstacle_points, obstacle_points[0]))
    for (x, y) in zip(obstacle_points[:,0], obstacle_points[:,1]):
        # x coordinate of the the occupied area
        ix = int(x)
        # y coordinate of the the occupied area
        iy = int(y)
        free_area = bresenham((prev_ix, prev_iy), (ix, iy))
        for fa in free_area:
            occupancy_map[fa[0]][fa[1]] = 0  # free area 0.0
        prev_ix = ix
        prev_iy = iy
    return occupancy_map

show_animation = False

def bresenham(start, end):
    """
    Implementation of Bresenham's line drawing algorithm
    See en.wikipedia.org/wiki/Bresenham's_line_algorithm
    Bresenham's Line Algorithm
    Produces a np.array from start and end (original from roguebasin.com)
    >>> points1 = bresenham((4, 4), (6, 10))
    >>> print(points1)
    np.array([[4,4], [4,5], [5,6], [5,7], [5,8], [6,9], [6,10]])
    """
    # setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)  # determine how steep the line is
    if is_steep:  # rotate line
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    # swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1  # recalculate differentials
    dy = y2 - y1  # recalculate differentials
    error = int(dx / 2.0)  # calculate error
    y_step = 1 if y1 < y2 else -1
    # iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = [y, x] if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx
    if swapped:  # reverse the list if the coordinates were swapped
        points.reverse()
    points = np.array(points)
    return points

def get_free_points_in_front(occupancy_grid, robot_pos, laser_range=10, num_laser=100):
    """
    Assumes circular lidar
    occupancy_grid: np.array (h x w)
    robot_pos: (x, y)

    Outputs: 
    free_points: np.array of hit points (x, y)
    """

    free_points = []
    hit_points = [] # actual hit points + last bresenham point (for some reason need this for flodding)
    actual_hit_points = [] # 
    for orientation in np.linspace(np.pi/2, 3*np.pi/2, num_laser):
        end_point = (round(robot_pos[0] + laser_range * np.cos(orientation)), round(robot_pos[1] + laser_range * np.sin(orientation)))
        
        # Get index along ray to check
        bresenham_points = (bresenham(robot_pos, end_point))

        # Go through the points and see the first hit
        # TODO: do a check if any first?
        for i in range(len(bresenham_points)):
            # if bresenham point is in the map 
            if bresenham_points[i,0] < 0 or bresenham_points[i,0] >= occupancy_grid.shape[0] or bresenham_points[i,1] < 0 or bresenham_points[i,1] >= occupancy_grid.shape[1]:
                if i != 0:
                    hit_points.append(bresenham_points[i-1])
                break # don't use this bresenham point 
            
            if occupancy_grid[bresenham_points[i,0], bresenham_points[i,1]] == 1: # hit if it is void or occupied #! THINK IF THIS IS A GOOD ASSUMPTION
                
                for j in range(min(4, len(bresenham_points) - i - 1)): # add 4 points in front of hit
                    free_points.append(bresenham_points[i+j])
                
                actual_hit_points.append(bresenham_points[i + j + 1])
                hit_points.append(bresenham_points[i + j + 1])
                
                break
            else: # no hits
                free_point = bresenham_points[i]
                free_points.append(free_point)

                if i == len(bresenham_points) - 1:
                    hit_points.append(end_point) # need to add this for proper flooding for vis mask
                    break
                
    
    # Convert to np.array
    free_points = np.array(free_points)
    hit_points = np.array(hit_points)
    actual_hit_points = np.array(actual_hit_points)
    return free_points, hit_points, actual_hit_points

if __name__ == "__main__":
    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, default="/path/to/raycast")
    parser.add_argument("--class_idx_building", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=60)
    parser.add_argument("--location", type=str, default="los_angeles")

    args = parser.parse_args()

    dataset_folder = Path(args.dataset_folder)
    bev_folder = dataset_folder / args.location / "semantic_masks"
    output_folder = dataset_folder / args.location / "flood_fill"

    output_folder.mkdir(exist_ok=True, parents=True)

    def generate_mask(filepath):
        mask = np.load(filepath)
        building_grid = mask[..., args.class_idx_building]
        try:
            flooded_image = get_raycast_building_mask(building_grid)
        except:
            raise Exception(f"Error in {filepath}")
        
        if flooded_image is not None:
            output_file = output_folder / filepath.name
            np.save(output_file, flooded_image)
        else:
            print("No flood fill generated")

    bev_files = list(bev_folder.iterdir())

    with Pool(args.num_workers) as p:
        for _ in tqdm(p.imap_unordered(generate_mask, bev_files), total=len(bev_files)):
            pass

