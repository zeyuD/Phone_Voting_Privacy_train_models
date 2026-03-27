from typing import Optional
import os
from multiprocessing import Pool, cpu_count
import glob
import re
import logging
from itertools import repeat, chain

import numpy as np
import pandas as pd
from tqdm import tqdm

from datasets import utils
import random
from variables import user_list, vote_list, user_othersetting_list

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
        self.target_user = config["target_user"]
        self.user_othersetting = config["user_othersetting"]
        self.user_othersetting_list = user_othersetting_list

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
        # Initialize data and labels
        # otherU_list = [u for u in user_list if u != self.target_user]
        if self.use == 'train':
            seg_data_1 = pd.read_csv(data_dir + vote_list[0] + '/' + user_list[0] + '_' + vote_list[0] + '_1.csv', header=None)
        else:
            seg_data_1 = pd.read_csv(data_dir + vote_list[0] + '/' + self.user_othersetting_list[0] + '_' + vote_list[0] + '_1.csv', header=None)
        num_dimensions = len(seg_data_1.columns)
        self.max_seq_len = len(seg_data_1)
        header_list = []
        for dim in range(0, num_dimensions):
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
            train_path = data_dir + vote
            test_path = data_dir + vote
            
            files = []
            if self.use == 'train':
                # for user in otherU_list:
                for user in user_list:
                    # Read number of files in path
                    user_files = [
                        file for file in os.listdir(train_path)
                        if user + "_" in file and not file.startswith('.') and
                        int(file.split('_')[-1].split('.')[0]) % 2 == start_idx
                    ]
                    files.extend(user_files)
                # # Random select x files for target user
                # target_user_files = [
                #     file for file in os.listdir(train_path)
                #     if self.target_user + "_" in file and not file.startswith('.') and
                #     int(file.split('_')[-1].split('.')[0]) % 2 == start_idx
                # ]
                # # target_user_files_train = random.sample(target_user_files, num_target_fewshot)
                # # files.extend(target_user_files_train)
            elif self.use == 'test':
                for user_othersetting in self.user_othersetting_list:
                    user_othersetting_files = [
                        file for file in os.listdir(test_path)
                        if user_othersetting + "_" in file and not file.startswith('.') and
                        int(file.split('_')[-1].split('.')[0]) % 2 == start_idx
                    ]
                    files.extend(user_othersetting_files)

            num_instance += len(files)
        labels = pd.DataFrame([0 for i in range(num_instance)], dtype=np.int32)
        users = pd.DataFrame(['' for i in range(num_instance)], dtype='object')

        for v in range(len(vote_list)):
            vote = vote_list[v]
            train_path = data_dir + vote
            test_path = data_dir + vote
            
            files = []
            if self.use == 'train':
                # for user in otherU_list:
                for user in user_list:
                    # print("Other user:", user)
                    # Read number of files in path
                    user_files = [
                        file for file in os.listdir(train_path)
                        if user + "_" in file and not file.startswith('.') and
                        int(file.split('_')[-1].split('.')[0]) % 2 == start_idx
                    ]
                    files.extend(user_files)
                # # Random select x files for target user
                # target_user_files = [
                #     file for file in os.listdir(train_path)
                #     if self.target_user + "_" in file and not file.startswith('.') and
                #     int(file.split('_')[-1].split('.')[0]) % 2 == start_idx
                # ]
                # # target_user_files_train = random.sample(target_user_files, num_target_fewshot)
                # # files.extend(target_user_files_train)
                print("Train files are in", train_path)
                print("Train files are", files)
                print("Number of train files:", len(files))
            elif self.use == 'test':
                for user_othersetting in self.user_othersetting_list:
                    user_othersetting_files = [
                        file for file in os.listdir(test_path)
                        if user_othersetting + "_" in file and not file.startswith('.') and
                        int(file.split('_')[-1].split('.')[0]) % 2 == start_idx
                    ]
                    files.extend(user_othersetting_files)
                print("Test files are in", test_path)
                print("Test files are", files)
                print("Number of test files:", len(files))
            for idx in range(0, len(files)):
                file = files[idx]
                if self.use == 'train':
                    seg_data_i = pd.read_csv(train_path + '/' + file, header=None)
                elif self.use == 'test':
                    seg_data_i = pd.read_csv(test_path + '/' + file, header=None)
                # if self.use == 'test':
                #     print("Test file:", file)
                # if seg_data_i.shape[0] != 100:
                #     print("Warning: Data shape mismatch for file", file)
                #     print("Data shape:", seg_data_i.shape)
                data_i = pd.DataFrame(dtype=np.float32, columns=header_list)
                for dim in range(0, num_dimensions):
                    data_i['dim_' + str(dim)] = seg_data_i[dim]
                data_i.index = [num_count] * len(data_i)
                data = pd.concat([data, data_i])
                labels.iloc[num_count] = v
                users.iloc[num_count] = self.target_user
                num_count += 1
        print("Num " + self.use + ": ", num_count)

        return data, labels, users

class VRSkeletonAtk(BaseData):
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

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):

        self.config = config

        # data_dir = root_dir.rsplit('/', 1)[0] + '/'
        # target_user = root_dir.rsplit('/', 1)[1]

        self.all_df, self.labels_df = self.load_all(root_dir, file_list=file_list, pattern=pattern)
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
        atk_ins = 8
        num_ins = 25*atk_ins

        # Initialize data and labels
        seg_data_1 = pd.read_csv(data_dir + '/' + '50.csv', header=None)
        num_dimensions = len(seg_data_1.columns)
        self.max_seq_len = len(seg_data_1)
        header_list = []
        for dim in range(0, num_dimensions):
            header_list.append('dim_' + str(dim))
        data = pd.DataFrame(dtype=np.float32, columns=header_list)
        labels = pd.DataFrame([0 for i in range(num_ins)],dtype=np.int32)

        num_count = 0
        # Add attacker data
        for ins in range(num_ins):
            for idx in range(ins*atk_ins, (ins+1)*atk_ins):
                if os.path.isfile(data_dir + '/' + str(idx+1) + '.csv'):
                    seg_data_i = pd.read_csv(data_dir + '/' + str(idx+1) + '.csv', header=None)
                    data_i = pd.DataFrame(dtype=np.float32, columns=header_list)
                    for dim in range(0, num_dimensions):
                        data_i['dim_' + str(dim)] = seg_data_i[dim]
                    data_i.index = [num_count] * len(data_i)
                    data = pd.concat([data, data_i])
                    labels.iloc[num_count] = 1
                    num_count += 1

        print("num all: ", num_count)
        labels = labels.iloc[:num_count]

        return data, labels


data_factory = {
                'vrs': VRSkeleton,
                'vrsatk': VRSkeletonAtk
                }
