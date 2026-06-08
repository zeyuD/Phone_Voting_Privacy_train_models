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
# from datasets_M import utils
from functions.detect_two_taps import detect_two_taps
from functions.band_pass_filter import band_pass_filter
from functions.crop_time import crop_time
from functions.interpolate_multiD import interpolate_multiD
import random
from run_fusion.variables_fuseFeat import user_list, vote_list, feature_names, interp_len

logger = logging.getLogger('__main__')


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows.

    Supports either:
        1. A single pd.DataFrame
        2. A dict of pd.DataFrame, e.g.
            {
                "eyeFeat": df_eye,
                "opticalFlow": df_flow,
            }

    For dict input, each modality has its own mean/std/min/max.
    """

    def __init__(self, norm_type, mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type:
                "none"
                "standardization"
                "minmax"
                "per_sample_std"
                "per_sample_minmax"

            mean, std, min_val, max_val:
                For single-dataframe mode:
                    pd.Series

                For multimodal dict mode:
                    dict[str, pd.Series]
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, data):
        """
        Args:
            data:
                pd.DataFrame
                or
                dict[str, pd.DataFrame]

        Returns:
            normalized data with the same structure as input
        """

        if self.norm_type == "none":
            return data

        if isinstance(data, dict):
            return self._normalize_dict(data)

        return self._normalize_df(data)

    def _normalize_dict(self, data_dict):
        """
        Normalize each modality dataframe separately.
        """

        normalized_dict = {}

        # Initialize statistics containers if needed
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = {}
            if self.std is None:
                self.std = {}

        elif self.norm_type == "minmax":
            if self.min_val is None:
                self.min_val = {}
            if self.max_val is None:
                self.max_val = {}

        for feature_name, df in data_dict.items():
            normalized_dict[feature_name] = self._normalize_df_for_key(
                df,
                feature_name
            )

        return normalized_dict

    def _normalize_df_for_key(self, df, key):
        """
        Normalize one dataframe inside a multimodal dict.
        Statistics are stored under self.mean[key], self.std[key], etc.
        """

        if self.norm_type == "none":
            return df

        elif self.norm_type == "standardization":
            if key not in self.mean:
                self.mean[key] = df.mean()
                self.std[key] = df.std()

            return (df - self.mean[key]) / (self.std[key] + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if key not in self.max_val:
                self.max_val[key] = df.max()
                self.min_val[key] = df.min()

            return (df - self.min_val[key]) / (
                self.max_val[key] - self.min_val[key] + np.finfo(float).eps
            )

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform("mean")) / (
                grouped.transform("std") + np.finfo(float).eps
            )

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform("min")
            max_vals = grouped.transform("max")

            return (df - min_vals) / (
                max_vals - min_vals + np.finfo(float).eps
            )

        else:
            raise NameError(f'Normalize method "{self.norm_type}" not implemented')

    def _normalize_df(self, df):
        """
        Original single-dataframe normalization behavior.
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

            return (df - self.min_val) / (
                self.max_val - self.min_val + np.finfo(float).eps
            )

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform("mean")) / (
                grouped.transform("std") + np.finfo(float).eps
            )

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform("min")
            max_vals = grouped.transform("max")

            return (df - min_vals) / (
                max_vals - min_vals + np.finfo(float).eps
            )

        else:
            raise NameError(f'Normalize method "{self.norm_type}" not implemented')


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
    Instead of concatenating all features into one dataframe, this version returns

    one dataframe per feature/modality:

        self.all_df_dict[feature_name]

    Each dataframe has:

        index = sample ID

        rows  = time steps

        cols  = dim_0, dim_1, ..., dim_k

    Attributes:
        all_df_dict: dataframe indexed by ID, with multiple rows corresponding to the same index (sample).
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


        self.all_df_dict, self.labels_df, self.users_df = self.load_all(
            data_dir,
            file_list=file_list,
            pattern=pattern
        )

        # Use the first feature to define sample IDs
        first_feature = feature_names[0]
        self.all_IDs = self.all_df_dict[first_feature].index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]

            # limit every modality dataframe
            for fname in feature_names:
                self.all_df_dict[fname] = self.all_df_dict[fname].loc[self.all_IDs]

            self.labels_df = self.labels_df.iloc[:limit_size]
            self.users_df = self.users_df.iloc[:limit_size]

        # use all features
        self.feature_names = feature_names
        # Optional: old single all_df points to first feature only
        # Better to update downstream code to use self.all_df_dict
        self.all_df = self.all_df_dict[first_feature]
        self.feature_df = self.all_df

        classes = self.labels_df[0].astype("category")
        self.class_names = classes.cat.categories


    def _read_crop_interp_one_file(self, filepath, feature_name=None):
        """
        Read one csv and process it by feature type.

        For normal features:
            detect taps -> keep original length -> set outside crop window to pad_value -> interpolate

        For no_crop_features:
            no crop/padding -> interpolate only

        Output:
            pd.DataFrame with shape [interp_len, feature_dim]
        """

        # -----------------------------
        # Config: features that skip crop
        # -----------------------------
        no_crop_features = None

        if isinstance(self.config, dict):
            no_crop_features = self.config.get("no_crop_features", None)

        if no_crop_features is None:
            no_crop_features = set()
        elif isinstance(no_crop_features, str):
            no_crop_features = {no_crop_features}
        else:
            no_crop_features = set(no_crop_features)

        # -----------------------------
        # Config: padding value
        # -----------------------------
        pad_value = 0.0

        if isinstance(self.config, dict):
            pad_value = self.config.get("crop_pad_value", 0.0)

        # -----------------------------
        # Read data
        # -----------------------------
        seg_data_i = pd.read_csv(filepath, header=None)

        # -----------------------------
        # Case 1: feature skips crop
        # -----------------------------
        if feature_name in no_crop_features:
            data_to_use_i = seg_data_i.copy()

        # -----------------------------
        # Case 2: soft crop with padding
        # -----------------------------
        else:
            _, centers_i = detect_two_taps(seg_data_i)

            valid_start = int(centers_i[0])
            valid_end = int(centers_i[1])

            # Safety clamp
            valid_start = max(0, min(valid_start, len(seg_data_i)))
            valid_end = max(valid_start, min(valid_end, len(seg_data_i)))

            data_to_use_i = seg_data_i.copy()

            # Apply band-pass filter if specified
            if self.config is not None and "pass_band" in self.config and self.config["pass_band"] is not None:
                low, high = self.config["pass_band"]
                data_to_use_i = band_pass_filter(data_to_use_i, low=low, high=high, fs=30)

            # Instead of removing outside region, pad it
            data_to_use_i.iloc[:valid_start, :] = pad_value
            data_to_use_i.iloc[valid_end:, :] = pad_value

        # -----------------------------
        # Interpolate to common length
        # -----------------------------
        data_to_use_i = interpolate_multiD(
            np.array(data_to_use_i, dtype=np.float64),
            interp_len
        )

        data_to_use_i = pd.DataFrame(data_to_use_i)

        return data_to_use_i


    def _get_common_files_for_user_vote(self, data_dir, user, vote):
        """
        Find files that exist in every feature folder for this user and vote.
        """
        common_files = None

        for feature_name in feature_names:
            folder = os.path.join(data_dir, feature_name, vote)
            files_f = [
                file for file in os.listdir(folder)
                if user + "_" in file and not file.startswith(".")
            ]

            if common_files is None:
                common_files = set(files_f)
            else:
                common_files = common_files & set(files_f)

        # Important: sort for deterministic train/test split
        common_files = sorted(list(common_files))

        return common_files
    

    def load_all(self, data_dir, file_list=None, pattern=None):
        """
        Returns:
            data_dict:
                {
                    feature_name_1: dataframe_for_feature_1,
                    feature_name_2: dataframe_for_feature_2,
                    ...
                }

            labels:
                pd.DataFrame, shape [num_samples, 1]

            users:
                pd.DataFrame, shape [num_samples, 1]
        """

        # --------------------------------------------------
        # Step 1: determine feature dimensions from one file
        # --------------------------------------------------
        feature_dims = {}
        self.max_seq_len = interp_len

        example_user = user_list[0]
        example_vote = vote_list[0]
        example_file = example_user + "_" + example_vote + "_1.csv"

        for feature_name in feature_names:
            example_path = os.path.join(
                data_dir,
                feature_name,
                example_vote,
                example_file
            )

            example_data = self._read_crop_interp_one_file(example_path, feature_name=feature_name)

            feature_dims[feature_name] = example_data.shape[1]

        # Create header list per feature
        header_dict = {}
        for feature_name in feature_names:
            num_dim = feature_dims[feature_name]
            header_dict[feature_name] = [
                f"{feature_name}_dim_{dim}" for dim in range(num_dim)
            ]

        # --------------------------------------------------
        # Step 2: decide train/test split
        # --------------------------------------------------
        if self.use == 'train':
            start_idx = 0
        elif self.use == 'test':
            start_idx = 1
        else:
            raise ValueError(f"Invalid use: {self.use}")

        # --------------------------------------------------
        # Step 3: first pass, collect valid sample metadata
        # --------------------------------------------------
        sample_meta = []

        for v, vote in enumerate(vote_list):
            for user in user_list:
                files = self._get_common_files_for_user_vote(data_dir, user, vote)

                data_idx = range(start_idx, len(files), 2)

                for idx in data_idx:
                    file = files[idx]

                    valid = True

                    # Check every feature for NaN after preprocessing
                    for feature_name in feature_names:
                        filepath = os.path.join(data_dir, feature_name, vote, file)

                        try:
                            data_to_use_i = self._read_crop_interp_one_file(filepath)
                        except Exception as e:
                            print(f"Warning: failed to read {filepath}: {e}")
                            valid = False
                            break

                        if data_to_use_i.isna().any().any():
                            print("Warning: NaN values in file", filepath)
                            valid = False
                            break

                    if valid:
                        sample_meta.append({
                            "vote_idx": v,
                            "vote": vote,
                            "user": user,
                            "file": file,
                        })

        num_instance = len(sample_meta)

        # --------------------------------------------------
        # Step 4: initialize one dataframe per feature
        # --------------------------------------------------
        data_dict = {}

        for feature_name in feature_names:
            data_dict[feature_name] = pd.DataFrame(
                dtype=np.float32,
                columns=header_dict[feature_name]
            )

        labels = pd.DataFrame([0 for _ in range(num_instance)], dtype=np.int32)
        users = pd.DataFrame(['' for _ in range(num_instance)], dtype='object')

        # --------------------------------------------------
        # Step 5: second pass, actually load data
        # --------------------------------------------------
        num_count = 0

        for meta in sample_meta:
            vote_idx = meta["vote_idx"]
            vote = meta["vote"]
            user = meta["user"]
            file = meta["file"]

            feature_data_this_sample = {}

            valid = True

            for feature_name in feature_names:
                filepath = os.path.join(data_dir, feature_name, vote, file)

                data_to_use_i = self._read_crop_interp_one_file(filepath, feature_name=feature_name)

                if data_to_use_i.isna().any().any():
                    print("Warning: NaN values in file", filepath)
                    valid = False
                    break

                feature_data_this_sample[feature_name] = data_to_use_i

            if not valid:
                continue

            # Add this sample to each modality dataframe
            for feature_name in feature_names:
                data_to_use_i = feature_data_this_sample[feature_name]

                data_i = pd.DataFrame(
                    dtype=np.float32,
                    columns=header_dict[feature_name]
                )

                for dim in range(feature_dims[feature_name]):
                    data_i[header_dict[feature_name][dim]] = data_to_use_i.iloc[:, dim]

                # sample index repeated for every time step
                data_i.index = [num_count] * len(data_i)

                data_dict[feature_name] = pd.concat(
                    [data_dict[feature_name], data_i],
                    axis=0
                )

            labels.iloc[num_count] = vote_idx
            users.iloc[num_count] = user

            num_count += 1

        # If some samples were skipped in the second pass, trim labels/users
        labels = labels.iloc[:num_count].reset_index(drop=True)
        users = users.iloc[:num_count].reset_index(drop=True)

        print("Num " + self.use + ": ", num_count)

        return data_dict, labels, users


data_factory = {
                'vrs': VRSkeleton
                }
