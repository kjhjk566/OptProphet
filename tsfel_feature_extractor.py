import pandas as pd
import tsfel
import numpy as np
import traceback
import warnings
from tqdm import tqdm
import os
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


def tsfel_windowed_features(data, window_size, step_size, colunm_name):
    """
    Extract TSFEL features using a sliding window approach.

    Parameters
    ----------
        data : pd.Series
            Time series data to extract features from.
        window_size : int
            Size of the sliding window.
        step_size : int
            Step size between successive windows.
        colunm_name : str
            Prefix used to rename the extracted feature columns.

    Returns
    -------
        pd.DataFrame
            Extracted feature matrix with column names prefixed by the input column name.
    """
    features = pd.DataFrame()
    cfg = tsfel.get_features_by_domain()
    for i in range(0, len(data), step_size):
        if i < window_size:
            continue
        else:
            window = data[i - window_size + 1:i + 1]
        feature = tsfel.time_series_features_extractor(cfg, window.to_numpy(),verbose=0)
        features = pd.concat([features, feature], axis=0)
    features = pd.DataFrame(features)
    features = features.rename(columns=lambda x: colunm_name + "_" + str(x))
    return features


def handle_raw_data(df, metric_used, window_size, step_size, dataset):
    """
    Extract features from raw metric data using TSFEL and align with original timestamps and labels.

    Parameters
    ----------
        df : pd.DataFrame
            Original input DataFrame containing raw time series and labels.
        metric_used : list of str
            List of metric column names to extract features from.
        window_size : int
            Size of the sliding window.
        step_size : int
            Step size between successive windows.
        dataset : str
            Dataset name identifier (e.g., 'offline', 'online').

    Returns
    -------
        pd.DataFrame
            DataFrame with original metrics, extracted features, timestamps, and labels.
    """
    all_features = pd.DataFrame()
    data = df[metric_used]
    for colunm in data.columns:
        f = tsfel_windowed_features(data[colunm], window_size, step_size, colunm)
        all_features = pd.concat([all_features, f], axis=1)
    if all_features.shape[0] < data.shape[0]:
        all_features = pd.concat(
            [pd.DataFrame(-1, index=range(data.shape[0] - all_features.shape[0]), columns=all_features.columns),
             all_features], axis=0)
    all_features = all_features.reset_index(drop=True)
    data = pd.concat([data, all_features], axis=1)
    data['timestamp'] = df['timestamp']
    data['subhealth_label'] = df['subhealth_label']
    data['failure_type'] = df['failure_type']
    data = data.iloc[window_size:]
    return data


def process_tsfel_data(metric_used, window_size, step_size, input_dirs, output_dirs, dataset):
    """
    Batch process TSFEL feature extraction for CSV files across multiple directories.

    Parameters
    ----------
        metric_used : list of str
            List of metric column names to extract features from.
        window_size : int
            Size of the sliding window.
        step_size : int
            Step size between successive windows.
        input_dirs : list of str
            List of directories containing input CSV files.
        output_dirs : list of str
            List of directories to save the output .npz files.
        dataset : str
            Dataset name identifier (e.g., 'offline', 'online').

    Returns
    -------
        None
            Processed files are saved as .npz files in the specified output directories.
    """
    for in_dir, out_dir in zip(input_dirs, output_dirs):
        os.makedirs(out_dir, exist_ok=True)
        for fname in tqdm(os.listdir(in_dir), desc=f"Processing {dataset} in {in_dir}"):
            try:
                path = os.path.join(in_dir, fname)
                df = pd.read_csv(path)
                processed = handle_raw_data(df, metric_used, window_size, step_size, dataset)
                arr = processed.to_numpy()
                out_name = os.path.splitext(fname)[0] + '.npz'
                np.savez(os.path.join(out_dir, out_name), data=arr)
            except Exception as e:
                traceback.print_exc()
                print(f"[Error] {fname}: {e}")

