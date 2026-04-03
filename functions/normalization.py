import numpy as np

def normalization(data, method='zscore'):
    """
    Normalize the data with different methods.
    Args:
        data: numpy array of shape (num_samples, num_features)
        method: normalization method, choose from 'zscore', etc.
    Returns:
        normalized_data: numpy array of shape (num_samples, num_features)
    """
    if method == 'zscore':
        # Normalize as a whole, not per dimension
        data_norm = (data - np.mean(data)) / (np.std(data) + np.finfo(float).eps)
    else:
        raise ValueError(f"Normalization method '{method}' not implemented.")
    
    return data_norm


# # Visualize to prove the normalization works
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     # Generate synthetic data
#     # 3D time series from small to large
#     # Each has different scale and mean
#     time_steps = 100
#     num_features = 3
#     data = np.zeros((time_steps, num_features))
#     for i in range(num_features):
#         data[:, i] = np.linspace(0, 10*(i+1), time_steps) + np.random.randn(time_steps) * (i+1) + (i+1)*5

#     # Normalize the data
#     normalized_data = normalization(data, method='zscore')

#     # Plot original and normalized data
#     fig, axs = plt.subplots(1, 3, figsize=(12, 5))
#     for i in range(num_features):
#         axs[i].plot(data[:, i], label='Original')
#         axs[i].plot(normalized_data[:, i], label='Normalized')
#         axs[i].set_title(f'Feature {i+1}')
#         axs[i].legend()
#     plt.tight_layout()
    
#     plt.show()