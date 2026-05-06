

def crop_time(df, crop_start, crop_end):
    # by default, use all time steps, time_axis = 0:len(df)
    time_axis = range(crop_start, len(df) - crop_end)
    cropped_df = df.iloc[time_axis]
    return cropped_df

# def crop_time(df, crop_start, crop_end):
#     # by default, use all time steps, time_axis = 0:len(df)
#     time_axis = range(crop_start, len(df) - crop_end)
#     cropped_df = df.iloc[time_axis]
#     return df
