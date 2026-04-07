import numpy as np

def interpolate_multiD(data, target_length):
    """
    Interpolates multi-dimensional data to a target length.
    
    Parameters:
    - data: A numpy array of shape (num_samples, num_dimensions).
    - target_length: The desired length of the output data.
    
    Returns:
    - A numpy array of shape (target_length, num_dimensions) containing the interpolated data.
    """
    num_samples, num_dimensions = data.shape
    interpolated_data = np.zeros((target_length, num_dimensions))
    
    for dim in range(num_dimensions):
        interpolated_data[:, dim] = np.interp(
            np.linspace(0, num_samples - 1, target_length),
            np.arange(num_samples),
            data[:, dim]
        )
    
    return interpolated_data