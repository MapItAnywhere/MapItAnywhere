<p align="center">
<h1 align="center">Map It Anywhere (MIA): Empowering Bird’s Eye View Mapping using Large-scale Public Data</h1>
<h3 class="is-size-5 has-text-weight-bold" style="color: orange;" align="center">
    NeurIPS 2024 Dataset and Benchmark Track
</h3>
  <p align="center">
    <a href="https://cherieho.com/"><strong>Cherie Ho*</strong></a>
    ·
    <a href="https://www.linkedin.com/in/tonyjzou/"><strong>Jiaye (Tony) Zou*</strong></a>
    ·
    <a href="https://www.linkedin.com/in/omaralama/"><strong>Omar Alama*</strong></a>
    <br>
    <a href="https://smj007.github.io/"><strong>Sai Mitheran Jagadesh Kumar</strong></a>
    ·
    <a href="https://github.com/chychiang"><strong>Benjamin Chiang</strong></a>
    ·
    <a href="https://www.linkedin.com/in/taneesh-gupta/"><strong>Taneesh Gupta</strong></a>
    ·
    <a href="https://sairlab.org/team/chenw/"><strong>Chen Wang</strong></a>
    <br>
    <a href="https://nik-v9.github.io/"><strong>Nikhil Keetha</strong></a>
    ·
    <a href="https://www.cs.cmu.edu/~./katia/"><strong>Katia Sycara</strong></a>
    ·
    <a href="https://theairlab.org/team/sebastian/"><strong>Sebastian Scherer</strong></a>
    <br>
  </p>

</p>
  <h3 align="center"><a href="https://arxiv.org/abs/2407.08726">Paper</a> | <a href="https://MapItAnywhere.github.io/">Project Page</a> | <a href="https://mapitanywhere-dataengine.hf.space">Data Engine Demo</a> | <a href="https://mapitanywhere-mapper.hf.space">Map Prediction Demo</a></h3>
  <div align="center"></div>

<p align="center">
  <img src="assets/mia_gif.gif" />
</p>


![Map It Anywhere (MIA)](/assets/mia_pull_fig.png "Map It Anywhere (MIA)")


## Table of Contents
  - [Get Your Own Map Prediction Data: Using the MIA Data Engine](#get-map-prediction-data-using-the-mia-data-engine)
  - [Use Our Map Prediction Data: Downloading the MIA dataset](#get-our-map-prediction-data-downloading-the-mia-dataset)
  - [Let's Predict Maps! Set up Mapper Env](#lets-predict-maps-setting-up-mapper-environment)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Acknowledgement](#acknowledgement)


## Get Map Prediction Data: Using the MIA data engine

### 0. Setting up the environment
0. Install docker by following the instructions on their [website](https://www.docker.com/get-started/)
1. Pull our docker image to set up the MIA data engine: 

        docker pull theairlab/mia:latest
2. Launch the container while mounting this repository to the container file system.

        docker run -v <PATH_TO_THIS_REPO>:/home/MapItAnywhere --network=host -it theairlab/mia:latest

### 1. Getting FPVs

The first stage of the MIA data engine is to get the first person view (FPV) images.
To get started, if you'd prefer a quick demonstration, you can use the example configuration `mia/conf/mia_quick.yaml`, which is set up with a few small areas to let you experience the full pipeline quickly. To use your own locations, copy the example configuration file from `mia/conf/example.yaml` and modify the cities list to include your desired locations.
Feel free to explore the other well-documented FPV options in the configuration file.

Second, you need to acquire an access token for the [Mapillary API](https://www.mapillary.com/developer/api-documentation).

Once configuration is done and you have your token simply run the following from inside your docker container with working dir set to this repo:

    cd /home/MapItAnywhere
    python3.9 -m mia.fpv.get_fpv --cfg mia/conf/<YOUR_CONFIG>.yaml --token <MLY_TOKEN>

That's it ! The engine will now automatically fetch, filter, and process your FPV images. You may get a few errors specifying that some images were unable to be fetched due to permission limitations. That is normal and the engine will continue.

Once all your locations have been downloaded, you will see that parquet files, images, and raw_images, have been populated in your `dataset_dir` for each location. You can now move on to getting BEVs.

### 2. Getting BEVs
Once you have the FPV parquet dataframes downloaded, you are now ready to fetch and generate the BEV smenatic maps. 

Edit the documented bev options in your configuration file to suit your use case. The defaults are tuned to what we used to produce the MIA datasets and you can use them as is.

You may also want to edit the stylesheet in `mia/bev/styles/mia.yml` used for rendering BEVs. Namely, the `driving_side` and `infer_sidewalks` options should be updated depending on the regions you are pulling from. For urbanized areas, set `infer_sidewalks=True`, for rural, set it to False. 

Once configuration is done simply run the following from inside your docker container with working dir set to this repo:

    python3.9 -m mia.bev.get_bev --cfg mia/conf/<YOUR_CONFIG>.yaml

The data engine will now fetch, process, and save the semantic masks.

You now have FPV-BEV pairs with associated metadata and camera parameters !

**Note** to get satellite imagery for comparison you must first download it by toggling the store_sat option in the configuration and setting up a google earth project.

### 3. (Optional) Visualize your data
You can visualize a few samples using the tool `mia/misc_tools/vis_samples.py`. 

From inside the container with working dir set to this repo, run:

    python3.9 -m mia.misc_tools.vis_samples --dataset_dir <PATH_TO_DATASET_SEE_CONFIG_YML> --locations <LOCATION_OF_INTEREST>

If successful, the script will generate a PDF called `compare.pdf` in the location directory. Upon openning you should see the metadata, FPVs, and BEVs of a few samples of the dataset. 


## Get Our Map Prediction Data: Downloading the MIA dataset
Refer to [mia/dataset.md](mia/dataset.md) for instructions.

## Let's Predict Maps! Setting Up Mapper Environment

### Install using pip
You can install all requirements using pip by running:

    pip install -r mapper/requirements.txt

### Use Docker
To use Mapper using Docker, please follow the steps:
1. Build the docker image `mapper/Dockerfile` by running: 
        
        cd mapper/
        docker build -t mapper:release mapper
2. Launch the container while mounting this repository to the container file system.
    
        docker run -v <PATH_TO_THIS_REPO>:/home/mapper --network=host -it --gpus=all mapper:release

## Training

### Pre-train with MIA Dataset
To pretrain using our paper configuration simply run:

    python -m mapper.mapper data.split=<PATH TO SPLIT FILE> data.data_dir=<PATH TO MIA DATASET>

### Finetune with NuScenes Dataset
To finetune using NuScenes Dataset with our paper configuration, run:

    python -m mapper.mapper -cn mapper_nuscenes training.checkpoint=<PATH TO PRETRAINED MODEL> data.data_dir=<PATH TO NUSCENES DATA> data.map_dir=<PATH TO GENERATED NUSCENES MAP>

## Reproduction
#### Dataset Setup
**MIA**: Follow download instructions in [Downloading the MIA Dataset](#downloading-the-mia-dataset)

**NuScenes**: Follow the data generation instructions in [Mono-Semantic-Maps](https://github.com/tom-roddick/mono-semantic-maps?tab=readme-ov-file#nuscenes). To match the newest available information, we use v1.3 of the NuScenes' map expansion pack. 

**KITTI360-BEV**: Follow the KITTI360-BEV dataset instructions in [SkyEye](https://github.com/robot-learning-freiburg/SkyEye?tab=readme-ov-file#skyeye-datasets)

#### Inference
To generate MIA dataset prediction results(on test split), use:

    python -m mapper.mapper data.split=<PATH TO SPLIT FILE> data.data_dir=<PATH TO MIA DATASET> training.checkpoint=<TRAINED WEIGHTS> training.eval=true
*To specify location, add `data.scenes` in the argument. For example, for held-out cities `data.scenes="[pittsburgh, houston]"`*

To Generate NuScenes dataset prediction results(on validation split), use:

    python -m mapper.mapper -cn mapper_nuscenes training.checkpoint=<PATH TO PRETRAINED MODEL> data.data_dir=<PATH TO NUSCENES DATA> data.map_dir=<PATH TO GENERATED NUSCENES MAP> training.eval=true

To Generate KITTI360-BEV dataset prediction results (on validation split), use:

    python -m mapper.mapper -cn mapper_kitti training.checkpoint=<PATH TO PRETRAINED MODEL> data.seam_root_dir=<PATH TO SEAM ROOT> data.dataset_root_dir=<PATH TO KITTI DATASET> training.eval=true

## Inference
We have also provided a script in case you want to map a custom image. To do so, first set up the environment, then run the following:

    python -m mapper.customized_inference training.checkpoint="<YOUR CHECKPOINT PATH>" image_path="<PATH TO YOUR IMAGE>" save_path="<PATH TO SAVE THE OUTPUT>"

## Trained Weights
We have hosted trained weights for Mapper model using MIA dataset on Huggingface. [Click Here](https://huggingface.co/mapitanywhere/mapper) to download.

## License
The FPVs were curated and processed from Mapillary and have the same CC by SA license. These include all images files, parquet dataframes, and dump.json. The BEVs were curated and processed from OpenStreetMap and has the same Open Data Commons Open Database (ODbL) License. These include all semantic masks and flood masks. The rest of the data is licensed under CC by SA license.

Code is licensed under CC by SA license.

## Acknowledgement
We thank the authors of the following repositories for their open-source code:
- [OrienterNet](https://github.com/facebookresearch/OrienterNet)
- [Map Machine](https://github.com/enzet/map-machine)
- [Mono-Semantic-Maps](https://github.com/tom-roddick/mono-semantic-maps)
- [Translating Images Into Maps](https://github.com/avishkarsaha/translating-images-into-maps)
- [SkyEye](https://github.com/robot-learning-freiburg/SkyEye)

## Citation

If you find our paper, dataset or code useful, please cite us:

```bib
@inproceedings{ho2024map,
title = {Map It Anywhere (MIA): Empowering Bird's Eye View Mapping using Large-scale Public Data},
author = {Ho, Cherie and Zou, Jiaye and Alama, Omar and Kumar, Sai Mitheran Jagadesh and Chiang, Benjamin and Gupta,
Taneesh and Wang, Chen and Keetha, Nikhil and Sycara, Katia and Scherer, Sebastian},
year = {2024},
booktitle = {Advances in Neural Information Processing Systems},
url = {https://arxiv.org/abs/2407.08726},
code = {https://github.com/MapItAnywhere/MapItAnywhere}
}
```