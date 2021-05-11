# Adapted from 3D PackNet
# https://github.com/TRI-ML/packnet-sfm/blob/master/packnet_sfm/datasets/kitti_dataset.py

import os
import glob
import random
from pathlib import Path
from operator import itemgetter
import itertools as it

import numpy as np
from torch.utils.data import Dataset
from einops import rearrange, reduce, repeat

from PIL import Image
# from torchvideotransforms import video_transforms

from .kitti_dataset_utils import load_oxts_packets_and_poses

########################################################################################################################

# Cameras from the stero pair (left is the origin)
IMAGE_FOLDER = {
    'left': 'image_02',
    'right': 'image_03',
}
DRIVE_ACTION = {
    'stop': 0.,
    'go': 1.,
}
# # Name of different calibration files
# CALIB_FILE = {
#     'cam2cam': 'calib_cam_to_cam.txt',
#     'velo2cam': 'calib_velo_to_cam.txt',
#     'imu2velo': 'calib_imu_to_velo.txt',
# }

OXTS_POSE_DATA = 'oxts'

# Change this to the directory where you store KITTI data
ROOT = Path('/content')
# KITTI_ROOT = ROOT / 'KittiRaw'
KITTI = 'KITTI' # assuming this is run 
KITTI_ROOT = ROOT / KITTI


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def pairwise(seq):
    a,b = tee(seq)
    b.next()
    return izip(a,b)


class RawDataset(Dataset):
    """
    KITTI Raw dataset class, containing list of images and corresponding odometry data
    This is passed into a Lightning datamodule
    This class defines the sequenced data for RGB images, and corresponding ground-truth OXTS velocity 
    Example format: /2011_09_28/2011_09_28_drive_0066_sync/<data type>
    Example data type: image_0<0-3>, oxts, velodyne_points
    Images: left-side RGB video images folder (image_02)
    Velocity: magnitude of vf: forward, and vl: leftward
    Training data is adjusted from test/val, as it contains training with priviledged OXTS data
    > Test ViT-pytorch vanilla (default, then larger images)
    > Training in batches /w pretrained also uses batched memory (online vs offline)
    > Offline sequence traininig (train batches of seq) vs online (train single seqs)
    > from monodepth2, use kitti_archives_to_download for all relevant RAW dataset file downloads
    > Train using full dataset (zhou pre-processed for removed static frames), no split, validation on other datasets
    > problem to tackle: how does it learn from events involving acceleration from braking in congestion? 
    >> For short time frames, ego will likely be stationary when front obstacles start moving, learning to accelerate later
    > Test method to extract all list of trainable files as list
    > Test using both camera images, and then using left only
    > use packnet dataset split files (zhou, test)
    > test with weak test-time supervision using pose network odometry outputs
    > test with smaller image sizes for memory
    > test by tokenizing patches or whole images
    > (640x192, from monodepth2 may be viable for this, gives 20x6 feature extraction space)
    > Test Smaller seq/Larger token size or Larger seq/smaller token size
    Note:
        You need to have downloaded the Kitti dataset first and provide the path to where it is saved.
        You can download the dataset using the script here: http://www.cvlibs.net/download.php?file=raw_data_downloader.zip

    Args:
        filenames: -may be unused > for all image file paths in the data split
        data_path: where to load the data from path, i.e. '/path/to/folder/with/kitti/'
        with_oxts: setting for return of stop/go data from oxts
        back_context: number of backward frames to consider as context, of interest for reading in past images
        forward_context: number of forward frames to consider as context, of interest for obtaining ground truth pose information in loss calculation
        strides: list of context strides (or frame differences in sequence building), 1 for no frame skip
        createcreate_data_list: cache list of data directories into text file, to speed up future runs
        img_ext
    """
    def __init__(
            self,
            # transform = None,
            # color_transform = None,
            train_args = None,
            root: str = ROOT,
            data_path: str = KITTI_ROOT,
            data_list_path: str = 'videowalk/code/data',
            with_oxts: bool = True,
            strides: tuple = (1,),
            create_data_list: bool = False,
            oxts_window_size: int = 2,
            include_curr_frame: bool = True,
            img_ext: str = '.jpg'
    ):
        super().__init__()

        # Assertions
        assert train_args.back_context >= 0 and train_args.forward_context >= 0, 'Invalid contexts'

        self.backward_context = train_args.back_context
        self.backward_context_paths = []
        self.forward_context = train_args.forward_context
        self.forward_context_paths = []

        self.with_context = (self.backward_context != 0 or self.forward_context != 0)
        self.with_oxts = with_oxts
        self.oxts_window_size = oxts_window_size

        self.root = Path(root)
        self.data_path = Path(data_path)
        self.data_list_path = self.root / data_list_path
        # self.transforms = transform
        # self.color_transform = color_transform
        self.train_args = train_args

        self.include_curr_frame = train_args.include_curr_frame
        self.augment_data = train_args.augment_data
        self.is_val = False

        self.img_ext = img_ext
        
        self._cache = {}
        self.oxts_cache = {} # used to store the values from a file

        self.loader = pil_loader
        self.DRIVE_ACTION = DRIVE_ACTION
        if self.train_args.label_smooth > 0.:
            stop_prob = 0. + train_args.label_smooth
            go_prob = 1. - train_args.label_smooth
            self.DRIVE_ACTION = {
                'stop': stop_prob,
                'go': go_prob,
            }

        # self.oxts_files = sorted(glob.glob(os.path.join(self.data_path, 'oxts', 'data', '*.txt')))

        # lists of directories of data
        self.left_paths = []
        self.right_paths = []
        self.oxts_paths = []
        
        # contain lists of dataset sample directories
        left_list_txt = self.data_list_path / 'image_02_list.txt'
        right_list_txt = self.data_list_path / 'image_03_list.txt'
        oxts_list_txt = self.data_list_path / 'oxts_list.txt'

        # collect list of all data files into a directory and append to text files, if needed
        if train_args.create_data_list:
            open(left_list_txt, 'w').close()
            open(right_list_txt, 'w').close()
            open(oxts_list_txt, 'w').close()
            # create lists txt for all data, and recursively search directories for sorted sample directories to append
            for left_sample in sorted(self.data_path.rglob('image_02/data/*.png')):
                with left_list_txt.open("a") as f:
                    # for each sample, remove the part containing the base directory, and add new line
                    f.write(str(left_sample))
                    f.write('\n')
            for right_sample in sorted(self.data_path.rglob('image_03/data/*.png')):
                with right_list_txt.open("a") as f:
                    f.write(str(right_sample))
                    f.write('\n')
            for oxts_sample in sorted(self.data_path.rglob('oxts/data/*.txt')):
                with oxts_list_txt.open("a") as f:
                    f.write(str(oxts_sample))
                    f.write('\n')
        
        # store txt directories for data in list
        # with open(left_list_txt, "r") as f:
        #     left_data = f.readlines()
        # with open(right_list_txt, "r") as f:
        #     right_data = f.readlines()
        # with open(oxts_list_txt, "r") as f:
        #     oxts_data = f.readlines()
        with open(left_list_txt, "r") as f:
            left_data = f.read().splitlines()
        with open(right_list_txt, "r") as f:
            right_data = f.read().splitlines()
        with open(oxts_list_txt, "r") as f:
            oxts_data = f.read().splitlines()

        # init list of paths for each category for easier access
        self.paths = []

        # Get file list from data
        for i, fname in enumerate(left_data):
            # path = str(ROOT / KITTI / fname)
            # self.left_paths.append(path)
            self.left_paths.append(fname)
        for i, fname in enumerate(right_data):
            self.right_paths.append(fname)
        for i, fname in enumerate(oxts_data):
            self.oxts_paths.append(fname)
        self.paths = self.left_paths # temp, test by training on main camera footage first
        self.paths.extend(self.right_paths)

        # If using context, filter file list
        if self.with_context: # true for this program
            paths_with_context = []
            for stride in strides: # single setting to start, can be used for multi-fps training
                for idx, file in enumerate(self.paths): # individual sample paths
                    backward_context_idxs, forward_context_idxs = \
                        self._get_sample_context(
                            file, self.backward_context, self.forward_context, stride) # return lists of ints
                    if backward_context_idxs is not None and forward_context_idxs is not None:
                        paths_with_context.append(self.paths[idx]) # add dataset path at index with fitting context
                        self.forward_context_paths.append(forward_context_idxs) # at matching dataset index, appends list of forward indices
                        self.backward_context_paths.append(backward_context_idxs[::-1]) # reverse indices (not pathnames)
            self.paths = paths_with_context # udpate to only include samples that have fitting contexts

    @staticmethod
    def _get_next_file(idx, file):
        """Get next file given specified idx and current file."""
        base, ext = os.path.splitext(os.path.basename(file))
        return os.path.join(os.path.dirname(file), str(idx).zfill(len(base)) + ext)
    
    @staticmethod
    def _get_oxts_file(image_file):
        """Gets the oxts file from an image file."""
        # find oxts pose file
        for cam in ['left', 'right']:
            # Check for both cameras, if found replace and return file name
            if IMAGE_FOLDER[cam] in image_file:
                return image_file.replace(IMAGE_FOLDER[cam], OXTS_POSE_DATA).replace('.png', '.txt')
        # Something went wrong (invalid image file)
        raise ValueError('Invalid KITTI path for pose supervision.')

    def _get_oxts_data(self, idx):
        """Gets the oxts data from an index."""
        oxts_file = self.oxts_paths[idx]
        if oxts_file in self.oxts_cache:
            oxts_data = self.oxts_cache[oxts_file]
        else:
            oxts_data = np.loadtxt(oxts_file, delimiter=' ', skiprows=0)
            self.oxts_cache[oxts_file] = oxts_data
        return oxts_data

########################################################################################################################
#### UTILITY METHODS FOR SEQUENCE
########################################################################################################################

    def _get_sample_context(self, sample_name,
                            backward_context, forward_context, stride=1):
        """
        Adapted from: https://github.com/TRI-ML/packnet-sfm/blob/master/packnet_sfm/datasets/kitti_dataset.py
        
        Get a sample context

        Parameters
        ----------
        sample_name : str
            Path + Name of the sample
        backward_context : int
            Size of backward context
        forward_context : int
            Size of forward context
        stride : int
            Stride value to consider when building the context

        Returns
        -------
        backward_context : list of int
            List containing the indexes for the backward context
        forward_context : list of int
            List containing the indexes for the forward context
        """
        # get sample video frame number and video folder
        base, ext = os.path.splitext(os.path.basename(sample_name))
        parent_folder = os.path.dirname(sample_name)
        f_idx = int(base)

        # Check number of files in folder
        if parent_folder in self._cache:
            max_num_files = self._cache[parent_folder]
        else:
            max_num_files = len(glob.glob(os.path.join(parent_folder, '*' + ext)))
            self._cache[parent_folder] = max_num_files

        # Check bounds
        if (f_idx - backward_context * stride) < 0 or (
                f_idx + forward_context * stride) >= max_num_files:
            return None, None

        # Backward context
        c_idx = f_idx
        backward_context_idxs = []
        while len(backward_context_idxs) < backward_context and c_idx > 0:
            c_idx -= stride
            filename = self._get_next_file(c_idx, sample_name)
            if os.path.exists(filename):
                backward_context_idxs.append(c_idx)
        if c_idx < 0:
            return None, None

        # Forward context
        c_idx = f_idx
        forward_context_idxs = []
        while len(forward_context_idxs) < forward_context and c_idx < max_num_files:
            c_idx += stride
            filename = self._get_next_file(c_idx, sample_name)
            if os.path.exists(filename):
                forward_context_idxs.append(c_idx)
        if c_idx >= max_num_files:
            return None, None

        return backward_context_idxs, forward_context_idxs

    def _get_context_files(self, sample_name, idxs):
        """
        Adapted from: https://github.com/TRI-ML/packnet-sfm/blob/master/packnet_sfm/datasets/kitti_dataset.py
 
        Returns image context files

        Parameters
        ----------
        sample_name : str
            Name of current sample
        idxs : list of idxs
            Context indexes

        Returns
        -------
        image_context_paths : list of str
            List of image names for the context
        """
        image_context_paths = [self._get_next_file(i, sample_name) for i in idxs]
        return image_context_paths

########################################################################################################################
#### DATASET ITERATION FUNCTIONS
########################################################################################################################

    def __len__(self):
        """Dataset length. Corresponds to data with viable contexts by default"""
        return len(self.paths)

    def __getitem__(self, idx):
        """Get dataset sample given an index."""
        # Add image information
        filename = self.paths[idx] # file of sample
        sample = {
            # 'idx': idx,
            # 'filename': filename,
            # 'rgb': self.transforms(pil_loader(filename)),
        }
        # Add context information if requested
        if self.with_context:
            # Add context images
            # all_context_idxs = self.backward_context_paths[idx] + \
            #                    self.forward_context_paths[idx]
            back_image_context_paths = \
                self._get_context_files(self.paths[idx], self.backward_context_paths[idx])
            if self.include_curr_frame:
                back_image_context_paths.append(filename)
            # create list of images for video_transform library
            clip_list = [pil_loader(f) for f in back_image_context_paths]
            # clip_list = [pil_loader(back_image_context_paths[0]), pil_loader(back_image_context_paths[-1])]
            # # augment data
            # if not self.is_val and self.augment_data and self.color_transform is not None:
            #     # rng to decide whether or not to augment color and flip
            #     color_aug = random.random() > 0.5
            #     # transform color
            #     if color_aug:
            #         image_context = self.color_transform(clip_list)
            #     else:
            #         image_context = self.transforms(clip_list)
            # else:
            #     image_context = self.transforms(clip_list)
            sample.update({
                'rgb_context': clip_list
            })
            # Add context poses
            if self.with_oxts:
                drive_action = self.DRIVE_ACTION
                # get list of forward context oxts paths from current idx
                forward_image_context_paths = \
                    self._get_context_files(self.paths[idx], self.forward_context_paths[idx])
                forward_paths_with_context = [filename]
                forward_paths_with_context.extend(forward_image_context_paths)
                oxts_paths_with_context = [self._get_oxts_file(forward_dir) for forward_dir in forward_paths_with_context]

                # get oxts packet tuples (packet[1] = (a,v))
                oxts_with_context = load_oxts_packets_and_poses(oxts_paths_with_context)

                # use moving average window (of variable size) to measure acceleration with lower noise
                # (check throttle with vel) and get differences for stop/go
                # throttle data length = frames + 1 - window
                throttle_data = []
                kinematics_data = np.asarray(oxts_with_context)
                acc = kinematics_data[:, 0]
                vel = kinematics_data[:, 1]
                if self.oxts_window_size == 2:
                    acc_pairwise = np.vstack((acc[:-1],acc[1:])) # shape = (2, n)
                    vel_pairwise = np.vstack((vel[:-1],vel[1:]))
                    # get the mean for each windowed values
                    acc_windowed = reduce(acc_pairwise, 'row sample -> sample', 'mean')
                    vel_windowed = reduce(vel_pairwise, 'row sample -> sample', 'mean')
                    # analyze each pair for each window
                    for a, v in zip(acc_windowed, vel_windowed):
                        if (a < -2 and v >= 2) or v < 2:
                            throttle_data.append(drive_action['stop']) # append drive_action stop
                        else:
                            throttle_data.append(drive_action['go']) # append drive_action go
                    sample.update({
                        'throttle': throttle_data
                    })
                else: 
                    return None


        # Return sample
        return sample

"""
    def __init__(self, data_root):
        self.samples = []

        for race in os.listdir(data_root):
            race_folder = os.path.join(data_root, race)

            for gender in os.listdir(race_folder):
                gender_filepath = os.path.join(race_folder, gender)

                with open(gender_filepath, 'r') as gender_file:
                    for name in gender_file.read().splitlines():
                        self.samples.append((race, gender, name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
"""