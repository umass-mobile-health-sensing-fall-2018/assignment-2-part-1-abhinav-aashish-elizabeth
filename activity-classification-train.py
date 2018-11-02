# -*- coding: utf-8 -*-
"""
This is the script used to train an activity recognition 
classifier on accelerometer data.

"""

import os
import sys
import numpy as np
from sklearn.tree import *
import sklearn.cross_validation
from sklearn.model_selection import cross_val_score
from features import extract_features
from util import slidingWindow, reorient, reset_vars
import pickle
from pprint import pprint
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


# %%---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

print("Loading data...")
sys.stdout.flush()
data_file = 'walking-data.csv'
data = np.genfromtxt(data_file, delimiter=',')
print("Loaded {} raw labelled activity data samples.".format(len(data)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                    Pre-processing
#
# -----------------------------------------------------------------------------

#print("Reorienting accelerometer data...")
sys.stdout.flush()
reset_vars()
reoriented = np.asarray([reorient(data[i,1], data[i,2], data[i,3]) for i in range(len(data))])
reoriented_data_with_timestamps = np.append(data[:,0:1],reoriented,axis=1)
data = np.append(reoriented_data_with_timestamps, data[:,-1:], axis=1)

# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

window_size = 20
step_size = 20

# sampling rate should be about 25 Hz; you can take a brief window to confirm this
n_samples = 1000
time_elapsed_seconds = (data[n_samples,0] - data[0,0]) / 1000
if(time_elapsed_seconds!=0):
    sampling_rate = n_samples / time_elapsed_seconds

# TODO: list the class labels that you collected data for in the order of label_index (defined in collect-labelled-data.py)
# DONE
class_names = ["walking", "sitting", "jumping", "running"]

#print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
sys.stdout.flush()

X = []
Y = []

for i,window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    window = window_with_timestamp_and_label[:,1:-1]   
    feature_names, x = extract_features(window)
    X.append(x)
    Y.append(window_with_timestamp_and_label[10, -1])

X = np.asarray(X)
Y = np.asarray(Y)
n_features = len(X)
    
print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(Y)))
print("\n")
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------


# TODO: split data into train and test datasets using 10-fold cross validation
# DONE
cv = sklearn.cross_validation.KFold(len(X), n_folds=10, shuffle=True, random_state=None)

"""
TODO: iterating over each fold, fit a decision tree classifier on the training set.
Then predict the class labels for the test set and compute the confusion matrix
using predicted labels and ground truth values. Print the accuracy, precision and recall
for each fold.
"""
accuracy= [] 
precision= []
recall= []
tree = DecisionTreeClassifier(criterion = "entropy",  max_depth = 3)
for train, test in cv:
    tree.fit(X[train], Y[train])
    y_pred = tree.predict(X[test])
    conf = sklearn.metrics.confusion_matrix(Y[test], y_pred)

# TODO: calculate and print the average accuracy, precision and recall values over all 10 folds
    accuracy.append(accuracy_score(Y[test], y_pred))
    precision.append(precision_score(Y[test], y_pred, average='weighted'))
    recall.append(recall_score(Y[test], y_pred, average='weighted'))

print("Accuracy", np.mean(accuracy))
print('Precision', np.mean(precision))
print('Recall', np.mean(recall))


# TODO: train the decision tree classifier on entire dataset

tree1 = tree.fit(X, Y)

# TODO: Save the decision tree visualization to disk - replace 'tree' with your decision tree and run the below line
export_graphviz(tree1, out_file='tree.dot', feature_names = feature_names)

# TODO: Save the classifier to disk - replace 'tree' with your decision tree and run the below line
with open('classifier.pickle', 'wb') as f:
    pickle.dump(tree1, f)