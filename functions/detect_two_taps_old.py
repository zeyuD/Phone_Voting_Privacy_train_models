from sklearn.cluster import KMeans
from scipy.signal import find_peaks
import numpy as np


def detect_two_taps(seg_data_f_1):

    plot_data = seg_data_f_1.copy()
    # Moving variance of each feature dimension
    feature_variances = seg_data_f_1.rolling(window=3).var()
    # There are multiple peaks in the variance, but 2 most significant peaks are correspond to the moments when the user tap the phone
    # The two peaks should not be too close to each other, and should be above a certain threshold
    # Estimate when they occur by checking the variance of each feature dimension, and find the time steps
    all_peaks = []
    for i in range(feature_variances.shape[1]):
        feature_variances_i = feature_variances.iloc[:, i]
        # Smoothing using filter to reduce noise
        feature_variances_i_smooth = feature_variances_i.rolling(window=3, center=True).mean()
        peaks, _ = find_peaks(feature_variances_i_smooth, distance=15, height=feature_variances_i_smooth.mean() * 0.6) # distance and height thresholds can be adjusted based on the data
        # Edge case: the end has only left side gradient, if so add the last time step as a peak if its variance is above the threshold
        if len(peaks) < 2:
            # print("Only detects " + str(len(peaks)) + " tap for feature " + str(i))
            # print("Last time step variance: " + str(feature_variances_i_smooth.iloc[len(feature_variances_i_smooth)-2]) + ", mean variance: " + str(feature_variances_i_smooth.mean()))
            # print("Last 5 time steps variance: " + str(feature_variances_i_smooth.iloc[-6:-1].tolist()))
            if feature_variances_i_smooth.iloc[len(feature_variances_i_smooth)-2] > feature_variances_i_smooth.iloc[-6:-1].mean():
                # print("Add last time step as a peak for feature")
                peaks = np.append(peaks, len(feature_variances_i_smooth) - 2)
        # print(f'Feature {i} peaks: {peaks.tolist()}')
        all_peaks.extend(peaks)
        plot_data.iloc[:, i] = feature_variances_i_smooth

    # all peaks should form two clusters, get 2 centers
    if len(all_peaks) >= 2:
        kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(all_peaks).reshape(-1, 1))
        centers = kmeans.cluster_centers_.flatten()
        # should be integer time steps, round to nearest int
        centers = np.round(centers).astype(int)
        # print(f'All peaks peak centers: {centers.tolist()}')
        # sort centers in ascending order
        centers.sort()

    return plot_data, centers