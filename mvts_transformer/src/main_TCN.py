# === src/main.py ===
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import os
from torch.utils.data import Dataset, DataLoader
from options import Options
from running import setup

from datasets.data import data_factory, Normalizer
from datasets.datasplit import split_dataset
from models.TCN_network import TCN

class GazeDataset(Dataset):
    def __init__(self, feature_df, label_df, id_list, max_seq_len):
        self.data = []
        self.labels = []
        for ID in id_list:
            seq = feature_df.loc[ID].values
            if seq.shape[0] > max_seq_len:
                seq = seq[:max_seq_len, :]
            else:
                pad_len = max_seq_len - seq.shape[0]
                seq = np.pad(seq, ((0, pad_len), (0, 0)), mode='constant')
            self.data.append(seq)
            self.labels.append(label_df.loc[ID].values[0])
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), self.labels[idx]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--data_class', type=str, default='vrs')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--normalization', type=str, default=None)
    parser.add_argument('--output_file', type=str, default='tcn_predictions.csv')
    return parser.parse_args()

def main(config):
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # config = vars(args)
    config['pattern'] = None
    config['n_proc'] = 4
    config['limit_size'] = None

    # Load training data
    data_class = data_factory[config['data_class']]
    my_data = data_class('train', config['data_dir'], pattern=config['pattern'], n_proc=config['n_proc'], limit_size=config['limit_size'], config=config)
    labels = my_data.labels_df.values.flatten()

    test_indices = None
    val_data = my_data
    val_indices = []

    if config['val_ratio'] > 0:
        train_indices, val_indices, test_indices = split_dataset(data_indices=my_data.all_IDs,
                                                                 validation_method='StratifiedShuffleSplit',
                                                                 n_splits=1,
                                                                 validation_ratio=config['val_ratio'],
                                                                 test_set_ratio=config['test_ratio'],
                                                                 test_indices=test_indices,
                                                                 random_seed=1337,
                                                                 labels=labels)
        train_indices = train_indices[0]
        val_indices = val_indices[0]
    else:
        train_indices = my_data.all_IDs
        if test_indices is None:
            test_indices = []

    print("Train indices: ", len(train_indices))
    print("Val indices: ", len(val_indices))

    test_data = data_class('test', config['data_dir'], pattern=config['pattern'], n_proc=config['n_proc'], limit_size=config['limit_size'], config=config)
    test_indices = test_data.all_IDs

    normalizer = None
    if config['normalization'] is not None:
        normalizer = Normalizer(config['normalization'])
        my_data.feature_df.loc[train_indices] = normalizer.normalize(my_data.feature_df.loc[train_indices])
        if len(val_indices):
            val_data.feature_df.loc[val_indices] = normalizer.normalize(val_data.feature_df.loc[val_indices])
        if len(test_indices):
            test_data.feature_df.loc[test_indices] = normalizer.normalize(test_data.feature_df.loc[test_indices])

    train_dataset = GazeDataset(my_data.feature_df, my_data.labels_df, train_indices, my_data.max_seq_len)
    val_dataset = GazeDataset(val_data.feature_df, val_data.labels_df, val_indices, my_data.max_seq_len)
    test_dataset = GazeDataset(test_data.feature_df, test_data.labels_df, test_indices, test_data.max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    num_classes = len(my_data.class_names)
    model = TCN(num_inputs=39, num_channels=[64]*3, num_classes=num_classes, kernel_size=3, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {total_loss:.4f}")

    model.eval()
    all_preds = []
    all_preds_scores = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            preds_scores = output.cpu().numpy()
            all_preds.extend(preds)
            all_preds_scores.extend(preds_scores)
            all_labels.extend(batch_y.numpy())

    all_preds_scores = np.array(all_preds_scores)
    A_score = all_preds_scores[:, 0]
    B_score = all_preds_scores[:, 1]
    C_score = all_preds_scores[:, 2]
    # print("A scores:", A_score)
    # print("B scores:", B_score)
    # print("C scores:", C_score)
    results_df = pd.DataFrame({'pred_labels': all_preds, 'target_labels': all_labels, 'score_A': A_score, 'score_B': B_score, 'score_C': C_score})
    results_df.to_csv(args.output_file, index=False)
    print(f"Results saved to {args.output_file}")

    # Show accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print("Test Accuracy:", accuracy)

if __name__ == '__main__':

    args = Options().parse()  # `argsparse` object
    config = setup(args)  # configuration dictionary
    main(config)