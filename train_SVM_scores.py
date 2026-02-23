import pandas as pd
import numpy as np
import os
import time
import sys
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'functions')))
from load_machine_config import load_machine_config


data_folder = "mvts_transformer/"
opti_feature = "opticalflow_44"
eye_feature = "all_processpos_norm"
suffix = "_downsample_480p_s22_5q"

# Read train scores (score_A to score_E) and labels from CSV files
opti_file = data_folder + "pred_labels_" + opti_feature + suffix + ".csv"
eye_file = data_folder + "pred_labels_" + eye_feature + suffix + ".csv"

opti_data = pd.read_csv(opti_file)
eye_data = pd.read_csv(eye_file)
target_labels = opti_data['target_labels'].tolist()
opti_pred_labels = opti_data['pred_labels'].tolist()
eye_pred_labels = eye_data['pred_labels'].tolist()
opti_scores = opti_data[['score_A', 'score_B', 'score_C', 'score_D', 'score_E']].values
eye_scores = eye_data[['score_A', 'score_B', 'score_C', 'score_D', 'score_E']].values

votes = ["A", "B", "C", "D", "E"]

#! Here is the main function
added_scores = opti_scores + eye_scores
opti_accuracy = accuracy_score(target_labels, opti_pred_labels)
eye_accuracy = accuracy_score(target_labels, eye_pred_labels)
print("Optical Flow Accuracy:", opti_accuracy)
print("Eye Feature Accuracy:", eye_accuracy)
added_max = np.argmax(added_scores, axis=1)
# print("Combined max:", combined_max)
added_accuracy = accuracy_score(target_labels, added_max)
added_f1 = f1_score(target_labels, added_max, average='weighted')
added_cm = confusion_matrix(target_labels, added_max)
print("Added Accuracy:", added_accuracy)
# Concatenate the optical flow and eye features
combined_scores = np.concatenate((opti_scores, eye_scores), axis=1)
# print("Combined train scores shape:", combined_scores.shape)

# Randomly split the data by 0.5 into train and test sets
x_train = []
y_train = []
x_test = []
y_test = []

random_seed = 42
random_idx = np.arange(len(target_labels))
np.random.seed(random_seed)

np.random.shuffle(random_idx)

split_point = len(target_labels) // 2
train_idx = random_idx[:split_point]
test_idx = random_idx[split_point:]

x_train = combined_scores[train_idx]
y_train = np.array(target_labels)[train_idx]
x_test = combined_scores[test_idx]
y_test = np.array(target_labels)[test_idx]

# print("Train set:", x_train)
# print("Test set:", x_test)
# print("Train labels:", y_train)
# print("Test labels:", y_test)

# Train SVM
svm = SVC(kernel='rbf', C=1.0, probability=True)
start_time = time.time()
svm.fit(x_train, y_train)
model_time = time.time()
print("Training time used:", model_time - start_time)

# Evaluate
y_pred = svm.predict(x_test)
# Scores for each class
y_scores = svm.decision_function(x_test)
# Save the scores to CSV
scores_df = pd.DataFrame(columns=['pred_labels', 'target_labels', 'score_A', 'score_B', 'score_C', 'score_D', 'score_E'])
scores_df['pred_labels'] = y_pred
scores_df['target_labels'] = y_test
for i, vote in enumerate(votes):
    scores_df['score_' + vote] = y_scores[:, i]
save_scores_file = data_folder + "pred_labels_SVM_" + opti_feature + "_" + eye_feature + suffix + ".csv"
scores_df.to_csv(save_scores_file, index=False)
test_time = time.time()
print("Testing time used:", test_time - model_time)

acc = accuracy_score(y_test, y_pred)
print("SVM Predicted Accuracy:", acc)
