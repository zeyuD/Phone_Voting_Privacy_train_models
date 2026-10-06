from typing import Optional
import os
from multiprocessing import Pool, cpu_count
import glob
import re
import logging
import sys
from itertools import repeat, chain

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory of this file to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'mvts_transformer_M')))
# from datasets import utils
from functions.detect_two_taps import detect_two_taps
from functions.crop_time import crop_time
from functions.interpolate_multiD import interpolate_multiD
import random
from variables_combineFeat import user_list, vote_list, feature_names, interp_len

logger = logging.getLogger('__main__')


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type, mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "none":
            return df
        
        elif self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            # print("grouped:", grouped.transform('std'))
            return (df - grouped.transform('mean')) / grouped.transform('std')

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y


def subsample(y, limit=256, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y


class BaseData(object):

    def set_num_processes(self, n_proc):

        if (n_proc is None) or (n_proc <= 0):
            self.n_proc = cpu_count()  # max(1, cpu_count() - 1)
        else:
            self.n_proc = min(n_proc, cpu_count())


class VRSkeleton(BaseData):
    """
    Dataset class for our own VR Skeleton dataset.
    Attributes:
        all_df: dataframe indexed by ID, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, use, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):
        self.use = use
        self.config = config

        data_dir = root_dir + '/'

        self.all_df, self.labels_df, self.users_df = self.load_all(data_dir, file_list=file_list, pattern=pattern)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        classes = self.labels_df[0].astype("category")
        self.class_names = classes.cat.categories
    
    def load_all(self, data_dir, file_list=None, pattern=None):
        # Read the same file from each feature's folder data_dir + feature_name + '/'
        # Concatenate the features as new columns in the dataframe
        
        seg_data_allFeat = pd.DataFrame()
        # Initialize data and labels
        for feature_name in feature_names:
            seg_data_f_1 = pd.read_csv(data_dir + feature_name + '/' + vote_list[0] + '/' + user_list[0] + '_' + vote_list[0] + '_1.csv', header=None)
            plot_data_1, centers_1 = detect_two_taps(seg_data_f_1)
            crop_start_1 = centers_1[0]
            crop_end_1 = len(seg_data_f_1) - centers_1[1]

            data_to_use_1 = seg_data_f_1
            # print("Original shape:", seg_data_f_1.shape)
            data_to_use_1 = crop_time(data_to_use_1, crop_start_1, crop_end_1)
            # print("Cropped shape:", seg_data_f_1.shape)
            data_to_use_1 = interpolate_multiD(np.array(data_to_use_1, dtype=np.float64), interp_len)
            data_to_use_1 = pd.DataFrame(data_to_use_1)
            seg_data_allFeat = pd.concat([seg_data_allFeat, data_to_use_1], axis=1)
        num_dimensions = len(seg_data_allFeat.columns)
        self.max_seq_len = len(seg_data_allFeat)
        header_list = []
        for dim in range(num_dimensions):
            header_list.append('dim_' + str(dim))
        data = pd.DataFrame(dtype=np.float32, columns=header_list)
        
        train_idx = [3, 4, 8, 9, 11, 12, 14, 16, 17, 18]
        # train_idx = random.sample(train_idx, 12)
        test_idx = [0, 1, 2, 5, 6, 7, 10, 13, 15, 19]
        num_other_ins = 5

        num_count = 0
        
        if self.use == 'train':
            start_idx = 0
        elif self.use == 'test':
            start_idx = 1
        else:
            print("Invalid use")

        # determine number of instances
        num_instance = 0
        for v in range(len(vote_list)):
            vote = vote_list[v]
            for user in user_list:
                feature_name_0 = feature_names[0]
                train_path = data_dir + feature_name_0 + '/' + vote
                files = [file for file in os.listdir(train_path) if user + "_" in file and not file.startswith('.')]
                for f in range(1, len(feature_names)):
                    feature_name = feature_names[f]
                    # Validate that the same name file exists in each feature folder
                    train_path = data_dir + feature_name + '/' + vote
                    files_f = [file for file in os.listdir(train_path) if user + "_" in file and not file.startswith('.')]
                    # Get the Intersection of files across features
                    files = list(set(files) & set(files_f))
                    
                data_idx = range(start_idx, len(files), 2)
                for idx in data_idx:
                    file = files[idx]
                    seg_data_i = pd.read_csv(train_path + '/' + file, header=None)
                    plot_data_i, centers_i = detect_two_taps(seg_data_i)
                    # if centers_i[0] == centers_i[1]:  # if only detects one tap, use the middle point as the second tap
                    #     print("Vote " + vote + ", user " + user + ", file " + file + ": only detects one tap")
                    # print("Centers detected at time steps:", centers_i)
                    crop_start_i = centers_i[0]
                    crop_end_i = len(seg_data_i) - centers_i[1]

                    data_to_use_i = seg_data_i
                    # print("Original shape:", seg_data_i.shape)
                    data_to_use_i = crop_time(data_to_use_i, crop_start_i, crop_end_i)
                    # print("Crop shape:", seg_data_i.shape)
                    data_to_use_i = interpolate_multiD(np.array(data_to_use_i, dtype=np.float64), interp_len)
                    data_to_use_i = pd.DataFrame(data_to_use_i)
                    data_i = pd.DataFrame(dtype=np.float32, columns=header_list)
                    # Print if has NaN values
                    if data_to_use_i.isna().any().any():
                        print("Warning: NaN values in file", file)
                        # Skip this instance
                        continue
                    num_instance += 1

        labels = pd.DataFrame([0 for i in range(num_instance)], dtype=np.int32)
        users = pd.DataFrame(['' for i in range(num_instance)], dtype='object')

        for v in range(len(vote_list)):
            vote = vote_list[v]
            train_path = data_dir + vote
            for user in user_list:
                feature_name_0 = feature_names[0]
                train_path = data_dir + feature_name_0 + '/' + vote
                files = [file for file in os.listdir(train_path) if user + "_" in file and not file.startswith('.')]
                for f in range(1, len(feature_names)):
                    feature_name = feature_names[f]
                    # Validate that the same name file exists in each feature folder
                    train_path = data_dir + feature_name + '/' + vote
                    files_f = [file for file in os.listdir(train_path) if user + "_" in file and not file.startswith('.')]
                    # Get the Intersection of files across features
                    files = list(set(files) & set(files_f))
                data_idx = range(start_idx, len(files), 2)
                for idx in data_idx:
                    file = files[idx]
                    seg_data_allFeat = pd.DataFrame()
                    for feature_name in feature_names:
                        seg_data_i = pd.read_csv(data_dir + feature_name + '/' + vote + '/' + file, header=None)
                        plot_data_i, centers_i = detect_two_taps(seg_data_i)
                        crop_start_i = centers_i[0]
                        crop_end_i = len(seg_data_i) - centers_i[1]

                        data_to_use_i = seg_data_i
                        data_to_use_i = crop_time(data_to_use_i, crop_start_i, crop_end_i)
                        data_to_use_i = interpolate_multiD(np.array(data_to_use_i, dtype=np.float64), interp_len)
                        data_to_use_i = pd.DataFrame(data_to_use_i)
                        seg_data_allFeat = pd.concat([seg_data_allFeat, data_to_use_i], axis=1)
                    # # if self.use == 'test':
                    # #     print("Test file:", file)
                    # if seg_data_i.shape[0] != 100:
                    #     print("Warning: Data shape mismatch for file", file)
                    #     print("Data shape:", seg_data_i.shape)
                    data_i = pd.DataFrame(dtype=np.float32, columns=header_list)
                    # Print if has NaN values
                    if seg_data_allFeat.isna().any().any():
                        print("Warning: NaN values in file", file)
                        # Skip this instance
                        continue
                    for dim in range(num_dimensions):
                        data_i['dim_' + str(dim)] = seg_data_allFeat.iloc[:, dim]
                    data_i.index = [num_count] * len(data_i)
                    data = pd.concat([data, data_i])
                    labels.iloc[num_count] = v
                    users.iloc[num_count] = user
                    num_count += 1
        print("Num " + self.use + ": ", num_count)

        return data, labels, users


data_factory = {
                'vrs': VRSkeleton
                }
