# -*- coding: utf-8 -*-
"""
This file is used for extracting features over windows of tri-axial accelerometer 
data. We recommend using helper functions like _compute_mean_features(window) to 
extract individual features.

As a side note, the underscore at the beginning of a function is a Python 
convention indicating that the function has private access (although in reality 
it is still publicly accessible).

"""

import numpy as np
from numpy import *
import scipy
from scipy.signal import find_peaks

def _compute_mean_features(window):
    """
    Computes the mean x, y and z acceleration over the given window. 
    """
    return np.mean(window, axis=0)

# TODO: define functions to compute more features

def _compute_variance_feature(window):
    """
    computes the variance
    """
    return np.var(window, axis=0)


def _compute_dominant_frequencies(window):
    """
    computes dominant frequencies
    """
    single_win = []
    for i in range(len(window)):
        r = math.sqrt(window[i][0] ** 2 + window[i][1] ** 2 + window[i][2] ** 2)
        single_win.append(r)
    fft = np.fft.rfft(single_win, axis=0)
    fft = fft.astype(float)
    
    return fft

def _calculate_peaks(window):
    """
    computes the number of peaks
    """
    single_win = []
    for i in range(len(window)):
        r = math.sqrt(window[i][0] ** 2 + window[i][1] ** 2 + window[i][2] ** 2)
        single_win.append(r)
    peaks = scipy.signal.find_peaks(single_win)[0]
    
    if(len(peaks) == 0):
        peaks = [0]
        return (window[peaks][0])
    return (window[peaks][0])


def extract_features(window):
    """
    Here is where you will extract your features from the data over 
    the given window. We have given you an example of computing 
    the mean and appending it to the feature vector.
    
    """
    
    x = []
    feature_names = []

    x.append(_compute_mean_features(window))
    feature_names.append("x_mean")
    feature_names.append("y_mean")
    feature_names.append("z_mean")

    # TODO: call functions to compute other features. Append the features to x and the names of these features to feature_names

    x.append(_compute_variance_feature(window))
    feature_names.append("x_variance")
    feature_names.append("y_variance")
    feature_names.append("z_variance")
    
    """
    returns multiple frequencies, so you need to loop through them and add
    appropriate number of labels to feature_names
    """
    frequencies = _compute_dominant_frequencies(window)
    x.append(frequencies)
    for i in range(len(frequencies)):
        feature_names.append("dominant frequencies")
    
    x.append(_calculate_peaks(window))
    feature_names.append("x_peaks")
    feature_names.append("y_peaks")
    feature_names.append("z_peaks")

    feature_vector = np.concatenate(x, axis=0) # convert the list of features to a single 1-dimensional vector
    return feature_names, feature_vector