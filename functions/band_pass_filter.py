import pandas as pd
from scipy.signal import butter, filtfilt


def band_pass_filter(seg_data_f_1, low, high, fs=30):

    # Apply band-pass filter to each feature dimension
    nyquist = 0.5 * fs

    if not (0 < low < high < nyquist):
        raise ValueError(
            f"Require 0 < low < high < Nyquist ({nyquist} Hz); got low={low}, high={high}."
        )

    b, a = butter(N=4, Wn=[low / nyquist, high / nyquist], btype='band')
    filtered = filtfilt(b, a, seg_data_f_1.values, axis=0)
    return pd.DataFrame(filtered, index=seg_data_f_1.index,
                        columns=seg_data_f_1.columns)