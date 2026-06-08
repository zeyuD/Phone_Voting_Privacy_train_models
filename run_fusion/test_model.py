"""
Modified by Zeyu Deng

Originally written by George Zerveas

Multimodal version:
- Data is stored in my_data.all_df_dict
- Each modality/feature has its own dataframe
- Normalizer supports dict[str, pd.DataFrame]
"""

import logging

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading packages ...")

import os
import sys
import time
import pickle
import json

import pandas as pd
import numpy as np

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix

# Add parent directory of this file to sys.path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'mvts_transformer_M', 'src')
    )
)

from options import Options
from running_M import setup, pipeline_factory, validate, check_progress, NEG_METRICS
from utils import utils
from datasets_M.datasplit import split_dataset
from models_M.ts_transformer import model_factory
from models_M.loss import get_loss_module
from optimizers import get_optimizer

from data_fuseFeat import data_factory, Normalizer
from variables_fuseFeat import vote_list, num_try

num_vote = len(vote_list)

def get_feature_dims(data_obj):
    """
    Return feature dimension for each modality.
    Example:
        {
            "eyeFeat": 39,
            "opticalFlow": 20
        }
    """
    return {
        feature_name: df.shape[1]
        for feature_name, df in data_obj.all_df_dict.items()
    }


def normalize_multimodal_data(
    my_data,
    val_data,
    test_data,
    train_indices,
    val_indices,
    test_indices,
    config,
):
    """
    Normalize multimodal data using the new Normalizer class.

    Assumes:
        my_data.all_df_dict:
            {
                "eyeFeat": DataFrame,
                "opticalFlow": DataFrame,
                ...
            }

    The normalizer stores one mean/std/min/max per modality.
    """

    normalizer = None

    if config['norm_from']:
        with open(config['norm_from'], 'rb') as f:
            norm_dict = pickle.load(f)

        normalizer = Normalizer(**norm_dict)

    elif config['normalization'] is not None:
        normalizer = Normalizer(config['normalization'])

        train_data_dict = {
            feature_name: df.loc[train_indices]
            for feature_name, df in my_data.all_df_dict.items()
        }

        normalized_train_dict = normalizer.normalize(train_data_dict)

        for feature_name in my_data.all_df_dict.keys():
            my_data.all_df_dict[feature_name].loc[train_indices] = normalized_train_dict[feature_name]

        if not config['normalization'].startswith('per_sample'):
            norm_dict = normalizer.__dict__

            with open(os.path.join(config['output_dir'], 'normalization.pickle'), 'wb') as f:
                pickle.dump(norm_dict, f, pickle.HIGHEST_PROTOCOL)

    if normalizer is not None:
        if len(val_indices):
            val_data_dict = {
                feature_name: df.loc[val_indices]
                for feature_name, df in val_data.all_df_dict.items()
            }

            normalized_val_dict = normalizer.normalize(val_data_dict)

            for feature_name in val_data.all_df_dict.keys():
                val_data.all_df_dict[feature_name].loc[val_indices] = normalized_val_dict[feature_name]

        if len(test_indices):
            test_data_dict = {
                feature_name: df.loc[test_indices]
                for feature_name, df in test_data.all_df_dict.items()
            }

            normalized_test_dict = normalizer.normalize(test_data_dict)

            for feature_name in test_data.all_df_dict.keys():
                test_data.all_df_dict[feature_name].loc[test_indices] = normalized_test_dict[feature_name]

    return normalizer


def build_input_dict(data_obj, indices, device=None):
    """
    Convert multimodal dataframe dictionary into a tensor dictionary.

    Output:
        {
            "eyeFeat": Tensor [B, T, D_eye],
            "opticalFlow": Tensor [B, T, D_flow],
            ...
        }
    """

    input_dict = {}

    for feature_name, df in data_obj.all_df_dict.items():
        modality_samples = []

        for sample_id in indices:
            x_i = df.loc[sample_id].values
            modality_samples.append(x_i)

        x = np.stack(modality_samples, axis=0)
        x = torch.tensor(x, dtype=torch.float32)

        if device is not None:
            x = x.to(device)

        input_dict[feature_name] = x

    return input_dict


def build_onehot_targets(labels_df, indices, num_classes):
    """
    Build one-hot targets from labels_df.

    Output:
        Tensor [B, num_classes]
    """

    targets = []

    for sample_id in indices:
        label_i = labels_df.iloc[sample_id].values
        label_i = int(label_i.item())

        target_i = [0] * num_classes
        target_i[label_i] = 1

        targets.append(target_i)

    targets = torch.tensor(targets, dtype=torch.float32)

    return targets


def main(config):

    total_start_time = time.time()

    file_handler = logging.FileHandler(os.path.join(config['output_dir'], 'output.log'))
    logger.addHandler(file_handler)

    logger.info('Running:\n{}\n'.format(' '.join(sys.argv)))

    if config['seed'] is not None:
        torch.manual_seed(config['seed'])

    device = config["device"]
    logger.info("Using device: {}".format(device))

    if device == 'cuda':
        logger.info("Device index: {}".format(torch.cuda.current_device()))

    # ---------------------------------------------------------
    # Build training data
    # ---------------------------------------------------------
    logger.info("Loading and preprocessing data ...")

    data_class = data_factory[config['data_class']]

    my_data = data_class(
        'train',
        config['data_dir'],
        pattern=config['pattern'],
        n_proc=config['n_proc'],
        limit_size=config['limit_size'],
        config=config,
    )

    # New multimodal feature dimensions
    feature_dims = get_feature_dims(my_data)
    config["feature_dims"] = feature_dims
    config["num_modalities"] = len(feature_dims)
    config["feat_dim"] = sum(feature_dims.values())  # optional backward compatibility

    logger.info("Feature dimensions per modality: {}".format(feature_dims))

    if config['task'] == 'classification':
        validation_method = 'StratifiedShuffleSplit'
        labels = my_data.labels_df.values.flatten()
    else:
        validation_method = 'ShuffleSplit'
        labels = None

    print("Data checking:")
    print("Feature dims:", feature_dims)
    print("Labels shape:", my_data.labels_df.shape)
    print("Number of IDs:", len(my_data.all_IDs))

    # ---------------------------------------------------------
    # Split dataset
    # ---------------------------------------------------------
    test_indices = None
    val_data = my_data
    val_indices = []

    if config['val_ratio'] > 0:
        train_indices, val_indices, test_indices = split_dataset(
            data_indices=my_data.all_IDs,
            validation_method=validation_method,
            n_splits=1,
            validation_ratio=config['val_ratio'],
            test_set_ratio=config['test_ratio'],
            test_indices=test_indices,
            random_seed=1337,
            labels=labels,
        )

        train_indices = train_indices[0]
        val_indices = val_indices[0]

    else:
        train_indices = my_data.all_IDs

        if test_indices is None:
            test_indices = []

    print("Train indices: ", train_indices)
    print("Test indices: ", test_indices)
    print("Val indices: ", val_indices)

    logger.info("{} samples may be used for training".format(len(train_indices)))
    logger.info("{} samples will be used for validation".format(len(val_indices)))
    logger.info("{} samples will be used for testing".format(len(test_indices)))

    with open(os.path.join(config['output_dir'], 'data_indices.json'), 'w') as f:
        try:
            json.dump(
                {
                    'train_indices': list(map(int, train_indices)),
                    'val_indices': list(map(int, val_indices)),
                    'test_indices': list(map(int, test_indices)),
                },
                f,
                indent=4,
            )
        except ValueError:
            json.dump(
                {
                    'train_indices': list(train_indices),
                    'val_indices': list(val_indices),
                    'test_indices': list(test_indices),
                },
                f,
                indent=4,
            )

    # ---------------------------------------------------------
    # Build explicit test data
    # ---------------------------------------------------------
    test_data = data_class(
        'test',
        config['data_dir'],
        pattern=config['pattern'],
        n_proc=config['n_proc'],
        limit_size=config['limit_size'],
        config=config,
    )

    test_indices = test_data.all_IDs

    # ---------------------------------------------------------
    # Normalize multimodal features
    # ---------------------------------------------------------
    normalizer = normalize_multimodal_data(
        my_data=my_data,
        val_data=val_data,
        test_data=test_data,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        config=config,
    )

    # ---------------------------------------------------------
    # Prepare result CSV files
    # ---------------------------------------------------------
    column_names = ['pred_labels', 'target_labels']

    for v in range(len(vote_list)):
        column_names.append('score_' + vote_list[v])

    df2 = pd.DataFrame(columns=column_names)
    df2.to_csv("train_pred_labels.csv", index=False)

    df3 = pd.DataFrame(columns=column_names)
    df3.to_csv("pred_labels.csv", index=False)

    df4 = pd.DataFrame(columns=vote_list)
    df4.to_csv("acc_avg.csv", index=False)

    # ---------------------------------------------------------
    # Training and testing loop
    # ---------------------------------------------------------
    for t in range(num_try):

        total_epoch_time = 0

        # -----------------------------------------------------
        # Create model
        # -----------------------------------------------------
        logger.info("Creating model ...")

        # model = model_factory(config, my_data)
        # Load last model (.pth) for testing
        model_para = torch.load(os.path.join(config['data_dir'], 'fusion_model.pth'))
        print(type(model_para))
        if isinstance(model_para, dict):
            print("Top-level keys:")
            print(model_para.keys())
            # Common cases
            if "state_dict" in model_para:
                state_dict = model_para["state_dict"]
            elif "model_state_dict" in model_para:
                state_dict = model_para["model_state_dict"]
            else:
                state_dict = model_para
        else:
            # If the full model object was saved
            print(model_para)
            state_dict = model_para.state_dict()
        print("\nLayers and parameter shapes:")
        for name, tensor in state_dict.items():
            if torch.is_tensor(tensor):
                print(f"{name:60s} {tuple(tensor.shape)}")

        # BRANCH = "opticalflowRAFT_edge_22_downsample_480p_s22"
        BRANCH = "all_processpos_norm_downsample_480p_s22"
        # Load parameters into model architecture
        model = model_factory(config, my_data)
        model_state = model.state_dict()
        partial_state = {}
        keep_prefixes = [
            f"project_inp.{BRANCH}",
            f"pos_enc.{BRANCH}",
            f"temporal_encoders.{BRANCH}",
            f"aux_output_layers.{BRANCH}",
        ]
        for k, v in state_dict.items():
            if any(k.startswith(prefix) for prefix in keep_prefixes):
                if k in model_state and model_state[k].shape == v.shape:
                    partial_state[k] = v
        model_state.update(partial_state)
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        print("Loaded tensors:")
        for k in partial_state:
            print(k)
        print("\nMissing keys:")
        print(missing)
        print("\nUnexpected keys:")
        print(unexpected)

        # Trim the model to only 
        model.to(device)

        # -----------------------------------------------------
        # Manual full test inference
        # -----------------------------------------------------
        num_vote = len(vote_list)

        model.eval()

        test_indices_list = list(test_indices)
        train_indices_list = list(my_data.all_IDs)

        input_dict = build_input_dict(
            data_obj=test_data,
            indices=test_indices_list,
            device=device,
        )

        train_input_dict = build_input_dict(
            data_obj=my_data,
            indices=train_indices_list,
            device=device,
        )

        targets = build_onehot_targets(
            labels_df=test_data.labels_df,
            indices=test_indices_list,
            num_classes=num_vote,
        )

        train_targets = build_onehot_targets(
            labels_df=my_data.labels_df,
            indices=train_indices_list,
            num_classes=num_vote,
        )

        padding_masks = torch.ones(
            (len(test_indices_list), model.max_len),
            dtype=torch.bool,
            device=device,
        )

        train_padding_masks = torch.ones(
            (len(train_indices_list), model.max_len),
            dtype=torch.bool,
            device=device,
        )

        with torch.no_grad():
            predictions = model(input_dict, padding_masks)
            train_predictions = model(train_input_dict, train_padding_masks)

        targets = targets.to(device)
        train_targets = train_targets.to(device)

        probs_p = predictions
        probs_t = targets

        predict_labels = []
        predict_scores = []
        target_labels = []

        train_predict_labels = []
        train_predict_scores = []
        train_target_labels = []

        for i in range(probs_p.shape[0]):
            predict_labels.append(torch.argmax(probs_p[i], dim=0).item())
            predict_scores.append(probs_p[i].detach().cpu().tolist())
            target_labels.append(torch.argmax(probs_t[i], dim=0).item())

        for j in range(train_predictions.shape[0]):
            train_predict_labels.append(torch.argmax(train_predictions[j], dim=0).item())
            train_predict_scores.append(train_predictions[j].detach().cpu().tolist())
            train_target_labels.append(torch.argmax(train_targets[j], dim=0).item())

        # -----------------------------------------------------
        # Save test predictions
        # -----------------------------------------------------
        predict_scores = np.array(predict_scores)

        pred_dict = {
            'pred_labels': predict_labels,
            'target_labels': target_labels,
        }

        for v, vote in enumerate(vote_list):
            pred_dict['score_' + vote] = predict_scores[:, v]

        df = pd.DataFrame(pred_dict)
        df.to_csv('pred_labels.csv', index=False)

        # -----------------------------------------------------
        # Save train predictions
        # -----------------------------------------------------
        train_predict_scores = np.array(train_predict_scores)

        train_pred_dict = {
            'pred_labels': train_predict_labels,
            'target_labels': train_target_labels,
        }

        for v, vote in enumerate(vote_list):
            train_pred_dict['score_' + vote] = train_predict_scores[:, v]

        train_df = pd.DataFrame(train_pred_dict)
        train_df.to_csv('train_pred_labels.csv', index=False)

        # -----------------------------------------------------
        # Confusion matrix and accuracy
        # -----------------------------------------------------
        cm = confusion_matrix(target_labels, predict_labels)
        cm_norm = confusion_matrix(target_labels, predict_labels, normalize='true')

        print(cm)
        print(cm_norm)

        acc = np.zeros((num_vote, 1))

        for i in range(num_vote):
            acc[i] = cm_norm[i][i]

        print("Average accuracy: ", acc.mean())

        df = pd.read_csv('acc_avg.csv')
        df.loc[df.shape[0], :] = acc.flatten()
        df.to_csv('acc_avg.csv', index=False)

        # -----------------------------------------------------
        # Confused instances
        # -----------------------------------------------------
        confused_instances = []
        confused_users = []
        messed_up_classes = []

        for i in range(len(predict_labels)):
            if predict_labels[i] != target_labels[i]:
                confused_instances.append(i)
                confused_users.append(test_data.users_df.iloc[i].values[0])
                messed_up_classes.append(predict_labels[i])

    # ---------------------------------------------------------
    # Save combined results from all tries
    # ---------------------------------------------------------
    df = pd.read_csv("acc_avg.csv")
    df5 = df.apply(lambda x: pd.Series(x.dropna().values))
    df5.to_csv("results/acc_avg_" + str(num_vote) + "q_fuseFeat.csv", index=False)

    print(df5)

    acc_avg = df5.mean().mean()
    print("Accuracy average:", acc_avg)

    df6 = pd.read_csv("pred_labels.csv")
    df6.to_csv("results/pred_labels_" + str(num_vote) + "q_fuseFeat.csv", index=False)

    df7 = pd.read_csv("train_pred_labels.csv")
    df7.to_csv("results/train_pred_labels_" + str(num_vote) + "q_fuseFeat.csv", index=False)


if __name__ == '__main__':

    args = Options().parse()
    config = setup(args)
    main(config)