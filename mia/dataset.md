# Map It Anywhere (MIA) Dataset

## Table of Contents
  - [Introduction](#introduction)
  - [Data](#data)
    - [Dataset Structure](#dataset-structure)
    - [Format](#format)
    - [Dataset Creation Summary](#dataset-creation)
  - [Getting Started](#getting-started)
  - [Licenses](#licenses)

![MIA Dataset Example](/assets/mia_dataset_overview.png "MIA Dataset Example")

![MIA Image Diversity](/assets/fpv_diversity.png "MIA Image Diversity")
## Introduction
The Map It Anywhere (MIA) dataset contains large-scale map-prediction-ready data curated from public datasets. 
Specifically, the dataset empowers Bird's Eye View (BEV) map prediction given First Person View (FPV) RGB images, by providing diversity in location and cameras beyond current datasets. The dataset contains 1.2 million high quality first-person-view (FPV) and bird's eye view (BEV) map pairs covering 470 squared kilometers, which to the best of our knowledge provides 6x more coverage than the closest publicly available map prediction dataset, thereby facilitating future map prediction research on generalizability and robustness. The dataset is curated using our MIA data engine [code](https://github.com/MapItAnywhere/MapItAnywhere) to sample from six urban-centered location: New York, Chicago, Houston, Los Angeles, Pittsburgh, and San Francisco.

## Data
### Dataset Structure

```
ROOT
|
--- LOCATION_0                             # location folder
|       |
|       +--- images                          # FPV Images (XX.jpg)
|       +--- semantic_masks                  # Semantic Masks (XX.npz)
|       +--- flood_fill                      # Visibility Masks (XX.npz)
|       ---- dump.json                       # Camera pose information for IDs in LOCATION
|       ---- image_points.parquet
|       ---- image_metadata.parquet
|       ---- image_metadata_filtered.parquet
|       ---- image_metadata_filtered_processed.parquet 
--- LOCATION_1                             
.
.
|
+-- LOCATION_2
--- README.md
--- samples.pdf # Visualization of sample data
```


## Format

Each data sample has a unique ID given by Mapillary and is used to reference and associate attributes related to the sample throughout the dataset.

**Each location has the following:**

- `images` Directory containing all FPV images named as `<id>_undistorted.png`
- `semantic_masks` npz files named as `<id>` containing semantic masks in the format of a single array `arr_0` with shape 224x224x8 where the 3rd dimension maps to classes as follows:
	0. road
	1. crossing
	2. explicit_pedestrian
	3. park (Unused by Mapper)
	4. building
	5. water (Unused by Mapper)
	6. terrain
	7. parking
	8. train (Unused by Mapper)
- `flood_masks` npz files named as `<id>` containing an observable region mask in the format of a single array `arr_0` with shape 224x224.
- `image_points.parquet` dataframe containing all image points retrieved within the tiles encompassing the boundary.
- `image_metadata.parquet` dataframe including metadata retrieved for each image point retrieved (After boundary filtering). The metadata retrieved is documented in the [Mapillary API](https://www.mapillary.com/developer/api-documentation#image)
- `image_metadata_filtered.parquet` As above but only keeping filtered records
- `image_metadata_filtered_processed.parquet` the final dataframe after FPV processing and spatial filtering and is the one that reflects what to expect in `images` directory.
- `dump.json` a json file containing camera intrinsics and extrinsics for each image taken. Same format as [OrienterNet](https://github.com/facebookresearch/OrienterNet).

In addition `split.json` is a file at the root that describes our training, validation, and testing splits.

**Note** that throughout the pipeline, some data samples are unable to be processed fully due to API issues or processing limitations. Such data samples may have residues in dataframes or split files but may not have corresponding maps or flood masks. Thus, a valid data sample is defined as one that has a corresponding image, metadata record, semantic mask, and flood mask. The invalid data samples are less than 0.001% and will be cleaned up in later versions.


## Dataset Creation

![MIA Curation Pipeline](/assets/mia_curation.png "MIA Curation Pipeline")
**Overview of how MIA data engine enables automatic curation of FPV & BEV data.**
Given names of cities as input from the left, the top row shows FPV processing, while the bottom row depicts BEV processing. Both pipelines converge on the right, producing FPV, BEV, and pose tuples. For more information, please reference the main paper.

### Curation Rationale

The MIA data engine and dataset were created to accelerate research progress towards anywhere map prediction. Current map prediction research builds on only a few map prediction datasets released by autonomous vehicle companies, which cover very limited area. We therefore present the MIA data engine, a more scalable approach by sourcing from large-scale crowd-sourced mapping platforms, Mapillary for FPV images and OpenStreetMap for BEV semantic maps. 


### Source Data

The MIA dataset includes data from two sources: [Mapillary](https://www.mapillary.com/) for First-Person-View (FPV) images, and [OpenStreetMap](https://www.openstreetmap.org) for Bird-Eye-View (BEV) maps. 

For FPV retrieval, we leverage Mapillary, a massive public database, licensed under [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/), with over 2 billion crowd-sourced images. The images span various weather and lighting conditions collected using diverse camera models and focal lengths. Furthermore, images are taken by pedestrians, vehicles, bicyclists, etc. This diversity enables the collection of more dynamic and difficult scenarios critical for anywhere map prediction.
When uploading to the Mapillary platform, users submit them under Mapillary's terms and all images shared are under a CC-BY-SA license, more details can be found in [Mapillary License Page](https://help.mapillary.com/hc/en-us/articles/115001770409-Licenses).
In addition, Mapillary integrates several mechanisms to minimize privacy concerns, such as applying technology to blur any faces and license plates, requiring users to notify if they observe any imageries that may contain personal data. More information can be found on the [Mapillary Privacy Policy page](https://www.mapillary.com/privacy).

For BEV retrieval, we leverage OpenStreetMap (OSM), a global crowd-sourced mapping platform open-sourced under [Open Data Commons Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/). OSM provides
rich vectorized annotations for streets, sidewalks, buildings, etc. OpenStreetMap has limitations on mapping private information where "it violates the privacy
of people living in this world", with guidelines found [here](https://wiki.openstreetmap.org/wiki/Limitations_on_mapping_private_information).


### Bias, Risks, and Limitations

While we show promising generalization performance on conventional datasets, we note that label noise inherently exists, to a higher degree 
than manually collected data, in crowd sourced data, in both pose correspondence, and in BEV map labeling. Such noise is common across large-scale
automatically scraped/curated benchmarks such as ImageNet. While we recognize that our sampled dataset is biased towards locations in the US, our MIA data engine is
applicable to other world-wide locations.
Our work relies heavily on crowd sourced data putting the burden of data collection on people and open-source contributions.


## Getting Started
1. [Download the dataset](https://cmu.box.com/s/6tnlvikg1rcsai0ve7t8kgdx9ago9x9q).
2. Unzip all locations of interest into the same structure described above, such that a root folder contains all location folders directly.
3. (Optional) Verify your download by visualizing a few samples using the tool `mia/misc_tools/vis_samples.py`. 
	1. Build the docker image `mia/Dockerfile` if you haven't already by running: 

		    docker build -t mia:release mia
	2. Launch the container while mounting your dataset root folder as well as this repository

			docker run -v <PATH_TO_DATASET_ROOT>:/home/mia_dataset_release -v <PATH_TO_THIS_REPO>:/home/MapItAnywhere --network=bridge -it mia:release
	3. From inside the container run:
		
		   cd /home/MapItAnywhere

           python3.9 -m mia.misc_tools.vis_samples --dataset_dir /home/mia_dataset_release --locations pittsburgh

	If successful, the script will generate a PDF called `compare.pdf` in the pittsburgh directory. Upon openning you should see the metadata, FPVs, and BEVs of a few samples of the dataset. Note that satellite imagery is not provided as part of the dataset and is only used for comparison purposes.

4. Enjoy and explore! Don't hesitate to raise a GitHub issue if you encounter any problems.

Samples and key metadata information in `compare.pdf` will look like the following:
![MIA Sample](/assets/sample_snippet.png "MIA Sample")

## Licenses
The FPVs were curated and processed from Mapillary and have the same [CC by SA license](https://creativecommons.org/licenses/by-sa/4.0/deed.en). These include all images files, parquet dataframes, and dump.json.
The BEVs were curated and processed from OpenStreetMap and has the same [Open Data Commons Open Database  (ODbL)  License](https://opendatacommons.org/licenses/odbl/). These include all semantic masks and flood masks.
The rest of the data is licensed under [CC by SA license](https://creativecommons.org/licenses/by-sa/4.0/deed.en).

