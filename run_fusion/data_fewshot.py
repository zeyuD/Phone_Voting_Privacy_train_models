"""
Data for the few-shot attacker (cust_run_fewshot.py): the attacker model of data_LOO.py plus
a few recordings of --target_user.

Train: every recording of every other user, plus the target user's even-position recordings
as the pool main_fewshot.py draws --num_fewshot shots per vote from. Test: the target user's
odd-position recordings, the same test files as in data_perUser.py and data_LOO.py.
"""

import os
import sys

import numpy as np
import pandas as pd

# Add parent directory of this file to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_fuseFeat import VRSkeleton
from run_fusion.variables_fuseFeat import user_list, vote_list, feature_names, interp_len, split_ref_features


class VRSkeletonFewshot(VRSkeleton):
    """VRSkeleton that loads only the users and recordings of this split, see users_and_halves()."""

    def users_and_halves(self):
        """
        Users to load, each with the part of its sorted recordings to use:
        'even' (positions 0, 2, ...), 'odd' (1, 3, ...) or 'all'.
        """
        target_user = self.config["target_user"]
        if target_user not in user_list:
            raise ValueError(f"--target_user {target_user} is not in user_list {user_list}")

        if self.use == "test":
            return [(target_user, "odd")]

        return [(user, "all") for user in user_list if user != target_user] + [(target_user, "even")]


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

            files:
                pd.DataFrame, shape [num_samples, 1], source file name of each sample
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
        if self.use not in ('train', 'test'):
            raise ValueError(f"Invalid use: {self.use}")

        users_and_halves = self.users_and_halves()

        # --------------------------------------------------
        # Step 3: first pass, collect valid sample metadata
        # --------------------------------------------------
        sample_meta = []

        for v, vote in enumerate(vote_list):
            for user, half in users_and_halves:
                files = self._get_common_files_for_user_vote(data_dir, user, vote)

                if half == "all":
                    data_idx = range(len(files))
                else:
                    data_idx = range(0 if half == "even" else 1, len(files), 2)

                for idx in data_idx:
                    file = files[idx]

                    valid = True

                    # Check every feature for NaN after preprocessing
                    for feature_name in feature_names:
                        filepath = self.feature_file_path(data_dir, feature_name, user, vote, file)

                        try:
                            data_to_use_i = self._read_crop_interp_one_file(filepath, feature_name=feature_name)
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
        if num_instance == 0:
            raise ValueError(
                f"No samples found for use='{self.use}'. No file exists in every folder of "
                f"feature_names + split_ref_features {list(dict.fromkeys(list(feature_names) + list(split_ref_features)))} "
                f"for users {[user for user, _ in users_and_halves]}; check that these folders contain these users."
            )

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
        files = pd.DataFrame(['' for _ in range(num_instance)], dtype='object')

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
                filepath = self.feature_file_path(data_dir, feature_name, user, vote, file)

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
            files.iloc[num_count] = file

            num_count += 1

        # If some samples were skipped in the second pass, trim labels/users
        labels = labels.iloc[:num_count].reset_index(drop=True)
        users = users.iloc[:num_count].reset_index(drop=True)
        files = files.iloc[:num_count].reset_index(drop=True)

        print("Num " + self.use + ": ", num_count)

        return data_dict, labels, users, files


data_factory = {
                'vrs': VRSkeletonFewshot
}
