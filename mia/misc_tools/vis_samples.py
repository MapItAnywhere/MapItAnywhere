import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
import pandas as pd
import numpy as np
import tqdm

from ..bev.get_bev import mask2rgb, PRETTY_COLORS as COLORS, VIS_ORDER
from ..fpv.filters import haversine_np, angle_dist

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", '-d', type=str, required=True, help="Dataset directory")
    parser.add_argument("--locations", '-l', type=str, default="all",
                        help="Location names in CSV format. Set to 'all' to traverse all locations.")
    parser.add_argument("--rows", type=int, default=5, help="How many samples per PDF page")
    parser.add_argument("--n_samples", type=int, default=30, help="How many samples to visualize?") 
    parser.add_argument("--store_sat", action="store_true", help="Add sattelite column") 
    args = parser.parse_args()

    MAX_ROWS = args.rows
    MAX_COLS = 4 if args.store_sat else 3
    MAX_TEXT_LEN=30
    
    locations = list()
    if args.locations.lower() == "all":
        locations = os.listdir(args.dataset_dir)
        locations = [l for l in locations if os.path.isdir(os.path.join(args.dataset_dir, l))]
    else:
        locations = args.locations.split(",")

    print(f"Parsing {len(locations)} locations..")

    all_locs_stats = dict()

    for location in tqdm.tqdm(locations):
        dataset_dir = Path(args.dataset_dir)
        location_dir = dataset_dir / location
        semantic_mask_dir = location_dir / "semantic_masks"
        sat_dir = location_dir / "sattelite"
        comp_dir = location_dir / "images"

        pq_name = 'image_metadata_filtered_processed.parquet'
        df = pd.read_parquet(location_dir / pq_name)

        # Calc derrivative attributes
        df["loc_descrip"] = haversine_np(
            lon1=df["geometry.long"], lat1=df["geometry.lat"],
            lon2=df["computed_geometry.long"], lat2=df["computed_geometry.lat"]
        )

        df["angle_descrip"] = angle_dist(
            df["compass_angle"],
            df["computed_compass_angle"]
        )

        with PdfPages(location_dir / 'compare.pdf') as pdf:
            # Plot legend page
            plt.figure()
            key2mask_i = dict(zip(COLORS.keys(), range(len(COLORS))))
            patches = [Patch(color=COLORS[key], label=f"{key}") for i,key in enumerate(VIS_ORDER) if COLORS[key] is not None]
            plt.legend(handles=patches, loc='center', title='Legend')
            plt.axis("off")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # Plot pairs
            row_cnt = 0
            fig = plt.figure(figsize=(MAX_COLS*2, MAX_ROWS*2))
            for index, row in tqdm.tqdm(df.iterrows()):
                id = row["id"]
                mask_fp = semantic_mask_dir / f"{id}.npz"
                comp_fp = comp_dir / f"{id}_undistorted.jpg"
                sat_fp = sat_dir / f"{id}.png"
                if not os.path.exists(mask_fp) or not os.path.exists(comp_fp) or \
                   (args.store_sat and not os.path.exists(sat_fp)):
                    continue
                plt.subplot(MAX_ROWS, MAX_COLS, (row_cnt % MAX_ROWS)*MAX_COLS + 1)
                plt.axis("off")
                desc = list()

                # Display attributes
                keys = ["geometry.long", "geometry.lat", "compass_angle",
                        "loc_descrip", "angle_descrip",
                        "make", "model", "camera_type", 
                        "quality_score"]
                for k in keys:
                    v = row[k]
                    if isinstance(v, float):
                        v = f"{v:.4f}"
                    bullet = f"{k}: {v}"
                    if len(bullet) > MAX_TEXT_LEN:
                        bullet = bullet[:MAX_TEXT_LEN-2] + ".."
                    desc.append(bullet)
                plt.text(0,0, "\n".join(desc), fontsize=7)
                plt.title(id)
                plt.subplot(MAX_ROWS, MAX_COLS, (row_cnt % MAX_ROWS)*MAX_COLS + 2)

                
                mask = np.load(mask_fp)["arr_0"]
                mask_rgb = mask2rgb(mask)
                plt.imshow(mask_rgb); plt.axis("off")
                plt.title(f"BEV")
                H,W,_ = mask_rgb.shape
                plt.scatter(np.array([H/2]), np.array([W/2]), marker="x")

                plt.subplot(MAX_ROWS, MAX_COLS, (row_cnt % MAX_ROWS)*MAX_COLS + 3)

                plt.imshow(plt.imread(comp_fp)); plt.axis("off")
                plt.title(f"FPV")

                if args.store_sat:
                    sat_fp = sat_dir / f"{id}.png"
                    plt.subplot(MAX_ROWS, MAX_COLS, (row_cnt % MAX_ROWS)*MAX_COLS + 4)
                    plt.imshow(plt.imread(sat_fp)); plt.axis("off")
                    plt.title(f"SAT")
                
                row_cnt += 1
                if row_cnt % MAX_ROWS == 0:
                    #plt.suptitle(location)
                    plt.tight_layout()
                    fig.align_titles()
                    pdf.savefig()
                    plt.close()
                    fig = plt.figure(figsize=(MAX_COLS*2, MAX_ROWS*2))
                
                if row_cnt == args.n_samples:
                    break