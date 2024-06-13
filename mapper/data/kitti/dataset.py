import os
import numpy as np
import torch.utils.data as data
import umsgpack
from PIL import Image
import json
import torchvision.transforms as tvf

from .transform import BEVTransform
from ..schema import KITTIDataConfiguration

class BEVKitti360Dataset(data.Dataset):
    _IMG_DIR = "img"
    _BEV_MSK_DIR = "bev_msk"
    _BEV_PLABEL_DIR = "bev_plabel_dynamic"
    _FV_MSK_DIR = "front_msk_seam"
    _BEV_DIR = "bev_ortho"
    _LST_DIR = "split"
    _PERCENTAGES_DIR = "percentages"
    _BEV_METADATA_FILE = "metadata_ortho.bin"
    _FV_METADATA_FILE = "metadata_front.bin"

    def __init__(self, cfg: KITTIDataConfiguration, split_name="train"):
        super(BEVKitti360Dataset, self).__init__()
        self.cfg = cfg
        self.seam_root_dir = cfg.seam_root_dir  # Directory of seamless data
        self.kitti_root_dir = cfg.dataset_root_dir  #  Directory of the KITTI360 data
        self.split_name = split_name
        
        self.rgb_cameras = ['front']
        if cfg.bev_percentage < 1:
            self.bev_percentage = cfg.bev_percentage
        else:
            self.bev_percentage = int(cfg.bev_percentage)

        # Folders
        self._img_dir = os.path.join(self.seam_root_dir, BEVKitti360Dataset._IMG_DIR)
        self._bev_msk_dir = os.path.join(self.seam_root_dir, BEVKitti360Dataset._BEV_MSK_DIR, BEVKitti360Dataset._BEV_DIR)
        self._bev_plabel_dir = os.path.join(self.seam_root_dir, BEVKitti360Dataset._BEV_PLABEL_DIR, BEVKitti360Dataset._BEV_DIR)
        self._fv_msk_dir = os.path.join(self.seam_root_dir, BEVKitti360Dataset._FV_MSK_DIR, "front")
        self._lst_dir = os.path.join(self.seam_root_dir, BEVKitti360Dataset._LST_DIR)
        self._percentages_dir = os.path.join(self.seam_root_dir, BEVKitti360Dataset._LST_DIR, BEVKitti360Dataset._PERCENTAGES_DIR)

        # Load meta-data and split
        self._bev_meta, self._bev_images, self._bev_images_all, self._fv_meta, self._fv_images, self._fv_images_all,\
        self._img_map, self.bev_percent_split = self._load_split()

        self.tfs = self.get_augmentations() if split_name == "train" else tvf.Compose([])
        self.transform = BEVTransform(cfg, self.tfs)

    def get_augmentations(self):

        print(f"Augmentation!", "\n" * 10)
        augmentations = [
            tvf.ColorJitter(
                brightness=self.cfg.augmentations.brightness,
                contrast=self.cfg.augmentations.contrast,
                saturation=self.cfg.augmentations.saturation,
                hue=self.cfg.augmentations.hue,
            )
        ]

        if self.cfg.augmentations.random_resized_crop:
            augmentations.append(
                tvf.RandomResizedCrop(scale=(0.8, 1.0))
            )  # RandomResizedCrop

        if self.cfg.augmentations.gaussian_noise.enabled:
            augmentations.append(
                tvf.GaussianNoise(
                    mean=self.cfg.augmentations.gaussian_noise.mean,
                    std=self.cfg.augmentations.gaussian_noise.std,
                )
            )  # Gaussian noise

        if self.cfg.augmentations.brightness_contrast.enabled:
            augmentations.append(
                tvf.ColorJitter(
                    brightness=self.cfg.augmentations.brightness_contrast.brightness_factor,
                    contrast=self.cfg.augmentations.brightness_contrast.contrast_factor,
                    saturation=0,  # Keep saturation at 0 for brightness and contrast adjustment
                    hue=0,
                )
            )  # Brightness and contrast adjustment

        return tvf.Compose(augmentations)
    
    # Load the train or the validation split
    def _load_split(self):
        with open(os.path.join(self.seam_root_dir, BEVKitti360Dataset._BEV_METADATA_FILE), "rb") as fid:
            bev_metadata = umsgpack.unpack(fid, encoding="utf-8")

        with open(os.path.join(self.seam_root_dir, BEVKitti360Dataset._FV_METADATA_FILE), 'rb') as fid:
            fv_metadata = umsgpack.unpack(fid, encoding="utf-8")

        # Read the files for this split
        with open(os.path.join(self._lst_dir, self.split_name + ".txt"), "r") as fid:
            lst = fid.readlines()
            lst = [line.strip() for line in lst]

        if self.split_name == "train":
            # Get all the frames in the train dataset. This will be used for generating samples for temporal consistency.
            with open(os.path.join(self._lst_dir, "{}_all.txt".format(self.split_name)), 'r') as fid:
                lst_all = fid.readlines()
                lst_all = [line.strip() for line in lst_all]

            # Get all the samples for which the BEV plabels have to be loaded.
            percentage_file = os.path.join(self._percentages_dir, "{}_{}.txt".format(self.split_name, self.bev_percentage))
            print("Loading {}% file".format(self.bev_percentage))
            with open(percentage_file, 'r') as fid:
                lst_percent = fid.readlines()
                lst_percent = [line.strip() for line in lst_percent]
        else:
            lst_all = lst
            lst_percent = lst

        # Remove elements from lst if they are not in _FRONT_MSK_DIR
        fv_msk_frames = os.listdir(self._fv_msk_dir)
        fv_msk_frames = [frame.split(".")[0] for frame in fv_msk_frames]
        fv_msk_frames_exist_map = {entry: True for entry in fv_msk_frames}  # This is to speed-up the dataloader
        lst = [entry for entry in lst if entry in fv_msk_frames_exist_map]
        lst_all = [entry for entry in lst_all if entry in fv_msk_frames_exist_map]

        # Filter based on the samples plabels
        if self.bev_percentage < 100:
            lst_filt = [entry for entry in lst if entry in lst_percent]
            lst = lst_filt

        # Remove any potential duplicates
        lst = set(lst)
        lst_percent = set(lst_percent)

        img_map = {}
        for camera in self.rgb_cameras:
            with open(os.path.join(self._img_dir, "{}.json".format(camera))) as fp:
                map_list = json.load(fp)
                map_dict = {k: v for d in map_list for k, v in d.items()}
                img_map[camera] = map_dict

        bev_meta = bev_metadata["meta"]
        bev_images = [img_desc for img_desc in bev_metadata["images"] if img_desc["id"] in lst]
        fv_meta = fv_metadata["meta"]
        fv_images = [img_desc for img_desc in fv_metadata['images'] if img_desc['id'] in lst]

        # Check for inconsistency due to inconsistencies in the input files or dataset
        bev_images_ids = [bev_img["id"] for bev_img in bev_images]
        fv_images_ids = [fv_img["id"] for fv_img in fv_images]
        assert set(bev_images_ids) == set(fv_images_ids) and len(bev_images_ids) == len(fv_images_ids), 'Inconsistency between fv_images and bev_images detected'

        if lst_all is not None:
            bev_images_all = [img_desc for img_desc in bev_metadata['images'] if img_desc['id'] in lst_all]
            fv_images_all = [img_desc for img_desc in fv_metadata['images'] if img_desc['id'] in lst_all]
        else:
            bev_images_all, fv_images_all = None, None

        return bev_meta, bev_images, bev_images_all, fv_meta, fv_images, fv_images_all, img_map, lst_percent

    def _find_index(self, list, key, value):
        for i, dic in enumerate(list):
            if dic[key] == value:
                return i
        return None

    def _load_item(self, item_idx):
        # Find the index of the element in the list containing all elements
        all_idx = self._find_index(self._fv_images_all, "id", self._fv_images[item_idx]['id'])
        if all_idx is None:
            raise IOError("Required index not found!")
        
        bev_img_desc = self._bev_images[item_idx]
        fv_img_desc = self._fv_images[item_idx]

        scene, frame_id = self._bev_images[item_idx]["id"].split(";")

        # Get the RGB file names
        img_file = os.path.join(
            self.kitti_root_dir,
            self._img_map["front"]["{}.png"
                                   .format(bev_img_desc['id'])]
        )

        if not os.path.exists(img_file):
            raise IOError(
                "RGB image not found! Scene: {}, Frame: {}".format(scene, frame_id)
            )

        # Load the images
        img = Image.open(img_file).convert(mode="RGB")

        # Load the BEV mask
        bev_msk_file = os.path.join(
            self._bev_msk_dir,
            "{}.png".format(bev_img_desc['id'])
        )
        bev_msk = Image.open(bev_msk_file)
        bev_plabel = None

        # Load the front mask
        fv_msk_file = os.path.join(
            self._fv_msk_dir, 
            "{}.png".format(fv_img_desc['id'])
        )
        fv_msk = Image.open(fv_msk_file)


        bev_weights_msk_combined = None

        # Get the other information
        bev_cat = bev_img_desc["cat"]
        bev_iscrowd = bev_img_desc["iscrowd"]
        fv_cat = fv_img_desc['cat']
        fv_iscrowd = fv_img_desc['iscrowd']
        fv_intrinsics = fv_img_desc["cam_intrinsic"]
        ego_pose = fv_img_desc['ego_pose']  # This loads the cam0 pose

        # Get the ids of all the frames
        frame_ids = bev_img_desc["id"]

        return img, bev_msk, bev_plabel, fv_msk, bev_weights_msk_combined, bev_cat, \
            bev_iscrowd, fv_cat, fv_iscrowd, fv_intrinsics, ego_pose, frame_ids

    @property
    def fv_categories(self):
        """Category names"""
        return self._fv_meta["categories"]

    @property
    def fv_num_categories(self):
        """Number of categories"""
        return len(self.fv_categories)

    @property
    def fv_num_stuff(self):
        """Number of "stuff" categories"""
        return self._fv_meta["num_stuff"]

    @property
    def fv_num_thing(self):
        """Number of "thing" categories"""
        return self.fv_num_categories - self.fv_num_stuff

    @property
    def bev_categories(self):
        """Category names"""
        return self._bev_meta["categories"]

    @property
    def bev_num_categories(self):
        """Number of categories"""
        return len(self.bev_categories)

    @property
    def bev_num_stuff(self):
        """Number of "stuff" categories"""
        return self._bev_meta["num_stuff"]

    @property
    def bev_num_thing(self):
        """Number of "thing" categories"""
        return self.bev_num_categories - self.bev_num_stuff

    @property
    def original_ids(self):
        """Original class id of each category"""
        return self._fv_meta["original_ids"]

    @property
    def palette(self):
        """Default palette to be used when color-coding semantic labels"""
        return np.array(self._fv_meta["palette"], dtype=np.uint8)

    @property
    def img_sizes(self):
        """Size of each image of the dataset"""
        return [img_desc["size"] for img_desc in self._fv_images]

    @property
    def img_categories(self):
        """Categories present in each image of the dataset"""
        return [img_desc["cat"] for img_desc in self._fv_images]

    @property
    def dataset_name(self):
        return "Kitti360"

    def __len__(self):
        if self.cfg.percentage < 1:
            return int(len(self._fv_images) * self.cfg.percentage)
        
        return len(self._fv_images)

    def __getitem__(self, item):
        img, bev_msk, bev_plabel, fv_msk, bev_weights_msk, bev_cat, bev_iscrowd, fv_cat, fv_iscrowd, fv_intrinsics, ego_pose, idx = self._load_item(item)
        
        rec = self.transform(img=img, bev_msk=bev_msk, bev_plabel=bev_plabel, fv_msk=fv_msk, bev_weights_msk=bev_weights_msk, bev_cat=bev_cat,
                             bev_iscrowd=bev_iscrowd, fv_cat=fv_cat, fv_iscrowd=fv_iscrowd, fv_intrinsics=fv_intrinsics,
                             ego_pose=ego_pose)
        size = (img.size[1], img.size[0])

        # Close the file
        img.close()
        bev_msk.close()
        fv_msk.close()

        rec["index"] = idx
        rec["size"] = size
        rec['name'] = idx 
        
        return rec

    def get_image_desc(self, idx):
        """Look up an image descriptor given the id"""
        matching = [img_desc for img_desc in self._images if img_desc["id"] == idx]
        if len(matching) == 1:
            return matching[0]
        else:
            raise ValueError("No image found with id %s" % idx)