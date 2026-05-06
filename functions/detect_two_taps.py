from sklearn.cluster import KMeans
from scipy.signal import find_peaks
import numpy as np
# from variables_combineFeat import crop_overlap
from variables_fuseFeat import crop_overlap

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
        peaks, _ = find_peaks(feature_variances_i_smooth, height=feature_variances_i_smooth.mean()) # height thresholds can be adjusted based on the data
        # Exclude those too close to the end
        peaks = [p for p in peaks if p < len(seg_data_f_1) - 15]
        all_peaks.extend(peaks)
        plot_data.iloc[:, i] = feature_variances_i_smooth
    if len(all_peaks) == 0:
    # If no peaks detected, use the 30
        print('No peaks detected, use default 30 as the tap moment')
        all_peaks = [30]

    # all peaks should form one clusters, get 1 center
    kmeans = KMeans(n_clusters=1, random_state=0).fit(np.array(all_peaks).reshape(-1, 1))
    centers = kmeans.cluster_centers_.flatten()
    # should be integer time steps, round to nearest int
    centers = np.round(centers).astype(int)
    # print(f'All peaks peak centers: {centers.tolist()}')

    # Add the last time step as a center
    centers = np.append(centers, len(seg_data_f_1) - 1)
    # sort centers in ascending order
    centers.sort()
    # All centers move backward by crop_overlap time steps to include the tap moment, but should not be less than 0
    centers = [c - crop_overlap if c - crop_overlap > 0 else 0 for c in centers]

    return plot_data, centers