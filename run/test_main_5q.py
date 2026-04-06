"""
Modified by Zeyu Deng

Originally written by George Zerveas

If you use any part of the code in this repository, please consider citing the following paper:
George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning, in
Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21), August 14--18, 2021
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

# 3rd party packages
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix

# Project modules
# Add parent directory of this file to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mvts_transformer', 'src')))

from options import Options
from running import setup, pipeline_factory, validate, check_progress, NEG_METRICS
from utils import utils
from test_data import data_factory, Normalizer
from datasets.datasplit import split_dataset
from models.ts_transformer import model_factory
from models.loss import get_loss_module
from optimizers import get_optimizer
from test_variables import user_list, vote_list
import matplotlib.pyplot as plt


def main(config):

    total_epoch_time = 0
    # total_eval_time = 0

    total_start_time = time.time()

    # Add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config['output_dir'], 'output.log'))
    logger.addHandler(file_handler)

    logger.info('Running:\n{}\n'.format(' '.join(sys.argv)))  # command used to run

    if config['seed'] is not None:
        torch.manual_seed(config['seed'])

    device = config["device"]
    logger.info("Using device: {}".format(device))
    if device == 'cuda':
        logger.info("Device index: {}".format(torch.cuda.current_device()))

    # Build data
    logger.info("Loading and preprocessing data ...")
    data_class = data_factory[config['data_class']]
    my_data = data_class('train', config['data_dir'], pattern=config['pattern'], n_proc=config['n_proc'], limit_size=config['limit_size'], config=config)
    feat_dim = my_data.feature_df.shape[1]  # dimensionality of data features
    if config['task'] == 'classification':
        validation_method = 'StratifiedShuffleSplit'
        labels = my_data.labels_df.values.flatten()
    else:
        validation_method = 'ShuffleSplit'
        labels = None

    # Data checking
    print("Data checking:")
    # Show data type, dimensions, and number of samples
    # print("Feature data type: ", my_data.feature_df.dtypes)
    # print("Feature dimensions: ", my_data.feature_df.shape)
    # print("Labels data type: ", my_data.labels_df.dtypes)
    # print("Labels dimensions: ", my_data.labels_df.shape)
    # print("Data: ", my_data)
    # print(labels)
    # print(len(labels))
    # print(my_data.all_IDs)
    # print(len(my_data.all_IDs))

    # Split dataset
    # test_data = my_data
    test_indices = None  # will be converted to empty list in `split_dataset`, if also test_set_ratio == 0
    val_data = my_data
    val_indices = []

    # Note: currently a validation set must exist, either with `val_pattern` or `val_ratio`
    # Using a `val_pattern` means that `val_ratio` == 0 and `test_ratio` == 0
    if config['val_ratio'] > 0:
        train_indices, val_indices, test_indices = split_dataset(data_indices=my_data.all_IDs,
                                                                 validation_method=validation_method,
                                                                 n_splits=1,
                                                                 validation_ratio=config['val_ratio'],
                                                                 test_set_ratio=config['test_ratio'],  # used only if test_indices not explicitly specified
                                                                 test_indices=test_indices,
                                                                 random_seed=1337,
                                                                 labels=labels)
        train_indices = train_indices[0]  # `split_dataset` returns a list of indices *per fold/split*
        val_indices = val_indices[0]  # `split_dataset` returns a list of indices *per fold/split*
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
            json.dump({'train_indices': list(map(int, train_indices)),
                       'val_indices': list(map(int, val_indices)),
                       'test_indices': list(map(int, test_indices))}, f, indent=4)
        except ValueError:  # in case indices are non-integers
            json.dump({'train_indices': list(train_indices),
                       'val_indices': list(val_indices),
                       'test_indices': list(test_indices)}, f, indent=4)

    test_data = data_class('test', config['data_dir'], pattern=config['pattern'], n_proc=config['n_proc'], limit_size=config['limit_size'], config=config)
    test_indices = test_data.all_IDs

    # Pre-process features
    normalizer = None
    if config['norm_from']:
        with open(config['norm_from'], 'rb') as f:
            norm_dict = pickle.load(f)
        normalizer = Normalizer(**norm_dict)
    elif config['normalization'] is not None:
        normalizer = Normalizer(config['normalization'])

        # Test normalization
        # print("Before normalization: ", my_data.feature_df.loc[train_indices])
        my_data.feature_df.loc[train_indices] = normalizer.normalize(my_data.feature_df.loc[train_indices])
        # print("After normalization: ", my_data.feature_df.loc[train_indices])

        if not config['normalization'].startswith('per_sample'):
            # get normalizing values from training set and store for future use
            norm_dict = normalizer.__dict__
            with open(os.path.join(config['output_dir'], 'normalization.pickle'), 'wb') as f:
                pickle.dump(norm_dict, f, pickle.HIGHEST_PROTOCOL)
    if normalizer is not None:
        if len(val_indices):
            val_data.feature_df.loc[val_indices] = normalizer.normalize(val_data.feature_df.loc[val_indices])
        if len(test_indices):
            test_data.feature_df.loc[test_indices] = normalizer.normalize(test_data.feature_df.loc[test_indices])

    # Create model
    logger.info("Creating model ...")
    model = model_factory(config, my_data)

    if config['freeze']:
        for name, param in model.named_parameters():
            if name.startswith('output_layer'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(utils.count_parameters(model)))
    logger.info("Trainable parameters: {}".format(utils.count_parameters(model, trainable=True)))


    # Initialize optimizer

    if config['global_reg']:
        weight_decay = config['l2_reg']
        output_reg = None
    else:
        weight_decay = 0
        output_reg = config['l2_reg']

    optim_class = get_optimizer(config['optimizer'])
    optimizer = optim_class(model.parameters(), lr=config['lr'], weight_decay=weight_decay)

    start_epoch = 0
    lr_step = 0  # current step index of `lr_step`
    lr = config['lr']  # current learning step
    # Load model and optimizer state
    if args.load_model:
        model, optimizer, start_epoch = utils.load_model(model, config['load_model'], optimizer, config['resume'],
                                                         config['change_output'],
                                                         config['lr'],
                                                         config['lr_step'],
                                                         config['lr_factor'])
    model.to(device)

    loss_module = get_loss_module(config)
    
    # Initialize data generators
    dataset_class, collate_fn, runner_class = pipeline_factory(config)
    val_dataset = dataset_class(val_data, val_indices)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=config['num_workers'],
                            pin_memory=True,
                            collate_fn=lambda x: collate_fn(x, max_len=model.max_len))

    train_dataset = dataset_class(my_data, train_indices)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              num_workers=config['num_workers'],
                              pin_memory=True,
                              collate_fn=lambda x: collate_fn(x, max_len=model.max_len))

    trainer = runner_class(model, train_loader, device, loss_module, optimizer, l2_reg=output_reg,
                                 print_interval=config['print_interval'], console=config['console'])
    val_evaluator = runner_class(model, val_loader, device, loss_module,
                                       print_interval=config['print_interval'], console=config['console'])

    tensorboard_writer = SummaryWriter(config['tensorboard_dir'])

    best_value = 1e16 if config['key_metric'] in NEG_METRICS else -1e16  # initialize with +inf or -inf depending on key metric
    metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    best_metrics = {}

    # Evaluate on validation before training
    aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config, best_metrics,
                                                          best_value, epoch=0)
    metrics_names, metrics_values = zip(*aggr_metrics_val.items())
    metrics.append(list(metrics_values))

    logger.info('Starting training...')
    for epoch in tqdm(range(start_epoch + 1, config["epochs"] + 1), desc='Training Epoch', leave=False):
        mark = epoch if config['save_all'] else 'last'
        epoch_start_time = time.time()
        aggr_metrics_train = trainer.train_epoch(epoch)  # dictionary of aggregate epoch metrics
        epoch_runtime = time.time() - epoch_start_time
        print()
        print_str = 'Epoch {} Training Summary: '.format(epoch)
        for k, v in aggr_metrics_train.items():
            tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)
        logger.info("Epoch runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(epoch_runtime)))
        total_epoch_time += epoch_runtime
        avg_epoch_time = total_epoch_time / (epoch - start_epoch)
        avg_batch_time = avg_epoch_time / len(train_loader)
        avg_sample_time = avg_epoch_time / len(train_dataset)
        logger.info("Avg epoch train. time: {} hours, {} minutes, {} seconds".format(*utils.readable_time(avg_epoch_time)))
        logger.info("Avg batch train. time: {} seconds".format(avg_batch_time))
        logger.info("Avg sample train. time: {} seconds".format(avg_sample_time))

        # evaluate if first or last epoch or at specified interval
        if (epoch == config["epochs"]) or (epoch == start_epoch + 1) or (epoch % config['val_interval'] == 0):
            aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config,
                                                                  best_metrics, best_value, epoch)
            metrics_names, metrics_values = zip(*aggr_metrics_val.items())
            metrics.append(list(metrics_values))

        # utils.save_model(os.path.join(config['save_dir'], 'model_{}.pth'.format(mark)), epoch, model, optimizer)
        utils.save_model(os.path.join(config['data_dir'], 'model_last.pth'), epoch, model)
        
        # Learning rate scheduling
        if epoch == config['lr_step'][lr_step]:
            # utils.save_model(os.path.join(config['save_dir'], 'model_{}.pth'.format(epoch)), epoch, model, optimizer)
            utils.save_model(os.path.join(config['data_dir'], 'model_last.pth'), epoch, model)
            lr = lr * config['lr_factor'][lr_step]
            if lr_step < len(config['lr_step']) - 1:  # so that this index does not get out of bounds
                lr_step += 1
            logger.info('Learning rate updated to: ', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Difficulty scheduling
        if config['harden'] and check_progress(epoch):
            train_loader.dataset.update()
            val_loader.dataset.update()

    # Export evolution of metrics over epochs
    header = metrics_names
    metrics_filepath = os.path.join(config["output_dir"], "metrics_" + config["experiment_name"] + ".xls")
    book = utils.export_performance_metrics(metrics_filepath, metrics, header, sheet_name="metrics")

    # Export record metrics to a file accumulating records from all experiments
    utils.register_record(config["records_file"], config["initial_timestamp"], config["experiment_name"],
                          best_metrics, aggr_metrics_val, comment=config['comment'])

    logger.info('Best {} was {}. Other metrics: {}'.format(config['key_metric'], best_value, best_metrics))
    logger.info('All Done!')

    total_runtime = time.time() - total_start_time
    logger.info("Total runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))

    # First, some target user's data
    # model = model.eval()
    # fps = 60
    # time_frag = 200
    # len_frag = math.floor(time_frag/1000 * fps)
    # overlap = 2
    num_vote = len(vote_list)

    
    input_0 = test_data.feature_df.loc[0].values
    label_0 = test_data.labels_df.iloc[0].values

    input_train_0 = my_data.feature_df.loc[0].values
    label_train_0 = my_data.labels_df.loc[0].values

    # print("Created labels: ", targets)
    input = torch.tensor(input_0, dtype=torch.float32)
    train_input = torch.tensor(input_train_0, dtype=torch.float32)
    # make input a batch of size 1
    input = input.unsqueeze(0)
    train_input = train_input.unsqueeze(0)
    targets = [[0]*num_vote]
    # targets[0][int(label_0)] = 1
    targets[0][int(label_0.item())] = 1
    train_targets = [[0]*num_vote]
    # train_targets[0][int(label_train_0)] = 1
    train_targets[0][int(label_train_0.item())] = 1
    for ins in range(1, len(test_indices)):
        # print("Processing instance:", ins)
        input_i = test_data.feature_df.loc[ins].values
        input_i = torch.tensor(input_i, dtype=torch.float32)
        # make input a batch of size 1
        input_i = input_i.unsqueeze(0)
        # print(ins, input.shape)
        # print(input_i.shape)
        input = torch.cat((input, input_i), dim=0)
        label_i = test_data.labels_df.iloc[ins].values
        target_i = [[0]*num_vote]
        # target_i[0][int(label_i)] = 1
        target_i[0][int(label_i.item())] = 1
        targets = np.concatenate((targets, target_i), axis=0)

    for all_ins in range(1, len(my_data.all_IDs)):
        train_input_i = my_data.feature_df.loc[all_ins].values
        train_input_i = torch.tensor(train_input_i, dtype=torch.float32)
        train_input_i = train_input_i.unsqueeze(0)
        train_input = torch.cat((train_input, train_input_i), dim=0)
        train_labels = my_data.labels_df.loc[all_ins].values
        train_target_i = [[0]*num_vote]
        # train_target_i[0][int(train_labels)] = 1
        train_target_i[0][int(train_labels.item())] = 1
        train_targets = np.concatenate((train_targets, train_target_i), axis=0)

    model.eval()
    # print(input)
    padding_masks = torch.ones((len(test_indices), model.max_len), dtype=torch.bool)
    train_padding_masks = torch.ones((len(my_data.all_IDs), model.max_len), dtype=torch.bool)
    # print(model(test_data.feature_df.values.to(device), padding_masks.to(device)))
    # print(padding_masks)
    # print(padding_masks.shape)
    predictions = model(input.to(device), padding_masks.to(device))
    targets = torch.tensor(targets, dtype=torch.float32)

    train_predictions = model(train_input.to(device), train_padding_masks.to(device))
    train_targets = torch.tensor(train_targets, dtype=torch.float32)

    # probs_p = torch.nn.functional.softmax(predictions, dim=0)
    # probs_t = torch.nn.functional.softmax(targets, dim=0)
    probs_p = predictions
    probs_t = targets
    predict_labels = []
    predict_scores = []
    target_labels = []
    # print(probs_p.shape)
    train_predict_labels = []
    train_predict_scores = []
    train_target_labels = []

    for i in range(probs_p.shape[0]):
        # print(probs_p[i])
        # print(torch.argmax(probs_p[i], dim=0))
        # # print(torch.argmax(probs_p[i], dim=1))
        # print(probs_t[i])
        # print(torch.argmax(probs_t[i], dim=0))
        # # print(torch.argmax(probs_t[i], dim=1))
        predict_labels.append(torch.argmax(probs_p[i], dim=0).tolist())
        predict_scores.append(probs_p[i].tolist())
        target_labels.append(torch.argmax(probs_t[i], dim=0).tolist())
    
    for j in range(train_predictions.shape[0]):
        train_predict_labels.append(torch.argmax(train_predictions[j], dim=0).tolist())
        train_predict_scores.append(train_predictions[j].tolist())
        train_target_labels.append(torch.argmax(train_targets[j], dim=0).tolist())

    # print(predict_labels)
    # print(target_labels)
    # print("Length of test labels",len(predict_labels))
    predict_scores = np.array(predict_scores)
    df = pd.read_csv('pred_labels.csv')
    # Add a new row at the end to the empty dataframe
    df = pd.DataFrame({'pred_labels': predict_labels, 'target_labels': target_labels, 'score_A': predict_scores[:, 0], 'score_B': predict_scores[:, 1],
                       'score_C': predict_scores[:, 2], 'score_D': predict_scores[:, 3], 'score_E': predict_scores[:, 4]})
    df.to_csv('pred_labels.csv', index=False)

    train_predict_scores = np.array(train_predict_scores)
    train_df = pd.read_csv('train_pred_labels.csv')
    train_df = pd.DataFrame({'pred_labels': train_predict_labels, 'target_labels': train_target_labels, 'score_A': train_predict_scores[:, 0], 'score_B': train_predict_scores[:, 1],
                             'score_C': train_predict_scores[:, 2], 'score_D': train_predict_scores[:, 3], 'score_E': train_predict_scores[:, 4]})
    train_df.to_csv('train_pred_labels.csv', index=False)

    # Confusion matrix
    print(confusion_matrix(target_labels, predict_labels))
    print(confusion_matrix(target_labels, predict_labels, normalize='true'))

    # Calculate accuracy for each class
    acc = np.zeros((num_vote, 1))
    for i in range(num_vote):
        acc[i] = confusion_matrix(target_labels, predict_labels, normalize='true')[i][i]

    # # # test_indices = range(len(test_data.labels_df.values))
    # test_dataset = dataset_class(test_data, test_indices)

    # test_loader = DataLoader(dataset=test_dataset,
    #                             batch_size=config['batch_size'],
    #                             shuffle=False,
    #                             num_workers=config['num_workers'],
    #                             pin_memory=True,
    #                             collate_fn=lambda x: collate_fn(x, max_len=model.max_len))
    # test_evaluator = runner_class(model, test_loader, device, loss_module,
    #                                     print_interval=config['print_interval'], console=config['console'])
    # aggr_metrics_test, per_batch_test, f1_avg, fnr_avg, fpr = test_evaluator.evaluate(keep_all=True)
    # print_str = 'Test Summary: '
    # # for k, v in aggr_metrics_test.items():
    # #     print(k, v)
    
    # print('Acc avg: {}'.format(acc.mean()))

    # Save test results
    df = pd.read_csv('acc_avg.csv')
    df.loc[df.shape[0], :] = acc.flatten()
    df.to_csv('acc_avg.csv', index=False)
    
    # df = pd.read_csv('FNR_avg.csv')
    # target_user = config['data_dir'].rsplit('/', 1)[1]
    # # print(df[target_user])
    # df.loc[df.shape[0], target_user] = fnr
    # df.to_csv('FNR_avg.csv', index=False)

    # Get confused instances and user names, and to which class they were predicted
    confused_instances = []
    confused_users = []
    messed_up_classes = []
    for i in range(len(predict_labels)):
        if predict_labels[i] != target_labels[i]:
            confused_instances.append(i)
            confused_users.append(test_data.users_df.iloc[i].values[0])
            messed_up_classes.append(predict_labels[i])
    # print(confused_instances)
    # print(confused_users)
    # print(messed_up_classes)

    return best_value


if __name__ == '__main__':

    args = Options().parse()  # `argsparse` object
    config = setup(args)  # configuration dictionary
    main(config)
