import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
import pandas as pd
import numpy as np
from tqdm import tqdm
import os


class MetricDataset(Dataset):
    """
    PyTorch Dataset for transforming multivariate time series data with chunking.

    Parameters
    ----------
        data : np.ndarray
            Input array of shape (time_steps, features).
        chunk_size : int, optional
            Length of each temporal chunk. Default is 20.
    """
    def __init__(self, data, chunk_size=20):
        reshaped_array = data.reshape(data.shape[0], 17, 138)
        final_array = np.transpose(reshaped_array, (0, 2, 1))
        self.original_len = data.shape[0]
        self.data = torch.from_numpy(final_array.astype(np.float32))
        self.chunk_size = chunk_size

        seq_len = self.data.shape[0]
        pad_size = (chunk_size - (seq_len % chunk_size)) % chunk_size

        if pad_size > 0:
            pad_tensor = torch.zeros(pad_size, self.data.shape[1], self.data.shape[2])
            self.padded_data = torch.cat([self.data, pad_tensor], dim=0)
        else:
            self.padded_data = self.data

        self.num_chunks = self.padded_data.shape[0] // chunk_size

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        """
        Extracts and chunks the time series for a given statistical feature index.

        Parameters
        ----------
            idx : int
                Index of the statistical feature to extract.

        Returns
        -------
            torch.Tensor
                Tensor of shape (num_chunks, chunk_size, feature_dim).
        """
        feature_data = self.padded_data[:, idx, :]
        chunked_data = feature_data.reshape(self.num_chunks, self.chunk_size, -1)
        return chunked_data


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder module for processing sequential data.

    Parameters
    ----------
        in_dim : int
            Input feature dimension.
        num_heads : int
            Number of attention heads.
        num_layers : int
            Number of Transformer encoder layers.
    """
    def __init__(self, in_dim, num_heads, num_layers):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=num_heads,
            dim_feedforward=in_dim * 2,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, features):
        """
        Forward pass for the Transformer encoder.

        Parameters
        ----------
            features : torch.Tensor
                Input tensor of shape (chunk_size, batch_size*num_chunks, feature_dim).

        Returns
        -------
            torch.Tensor
                Encoded tensor of the same shape as input.
        """
        h = F.leaky_relu(self.encoder(features))
        return h


def train(file_dir, transformer_file_dir):
    """
    Train the Transformer encoder on time-series feature data and save encoded outputs.

    Parameters
    ----------
        file_dir : str
            Directory containing input .npz files.
        transformer_file_dir : str
            Directory to save encoded output .npz files.
    """
    batch_size = 64
    chunk_size = 20
    cols_with_nan_indices = [41, 86, 178, 223, 315, 360, 452, 497, 589, 634, 726, 771, 863, 908, 1000,
                             1045, 1137, 1182, 1274, 1319, 1411, 1456, 1548, 1593, 1685, 1730, 1822, 1867,
                             1959, 2004, 2096, 2141, 2233, 2278]

    for file in tqdm(os.listdir(file_dir)):
        with np.load(file_dir + file, allow_pickle=True) as offline_data:
            print(file_dir + file)
            X_train = offline_data['data'][:, :-3]
            label = offline_data['data'][:, -2:]
            print(X_train.shape)

            X_train = np.array(X_train, dtype=np.float32)

        dataset = MetricDataset(X_train, chunk_size=chunk_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        transformer = TransformerEncoder(in_dim=17, num_heads=1, num_layers=2)
        all_features = []

        for batch in tqdm(dataloader):
            """
            Reshape batch for Transformer input and collect outputs.

            batch.shape : (batch_size, num_chunks, chunk_size, feature_dim)
            Transformed to: (chunk_size, batch_size*num_chunks, feature_dim)
            """
            batch_size, num_chunks, chunk_size, feature_dim = batch.shape
            batch = batch.view(batch_size * num_chunks, chunk_size, feature_dim)
            batch = batch.permute(1, 0, 2)
            output = transformer(batch)
            output = output.permute(1, 0, 2).view(batch_size, num_chunks, chunk_size, feature_dim)
            all_features.append(output)

        final_output = torch.cat(all_features, dim=0)
        final_output = final_output.reshape(138, -1, 17)
        final_output = final_output.permute(1, 2, 0)
        flattened_data = final_output.reshape(final_output.shape[0], -1)
        flattened_data = flattened_data[:dataset.original_len]
        demo_npz = np.hstack((flattened_data.detach().numpy(), label))

        if X_train.shape[0] != demo_npz.shape[0] or demo_npz.shape[1] != 2312:
            print(X_train.shape, demo_npz.shape)

        np.savez(transformer_file_dir + file, data=demo_npz)


def full_transformer_workflow(
    offline_tsfel_dirs,
    offline_feature_dirs,
    online_tsfel_dirs,
    online_feature_dirs,
    stacked_output_path
):
    """
    Execute full Transformer-based feature extraction and offline feature stacking.

    Parameters
    ----------
        offline_tsfel_dirs : list of str
            Directories containing offline TSFEL features.
        offline_feature_dirs : list of str
            Directories to save offline Transformer-encoded features.
        online_tsfel_dirs : list of str
            Directories containing online TSFEL features.
        online_feature_dirs : list of str
            Directories to save online Transformer-encoded features.
        stacked_output_path : str
            File path to save the stacked offline feature output.
    """
    for in_dir, out_dir in zip(offline_tsfel_dirs, offline_feature_dirs):
        os.makedirs(out_dir, exist_ok=True)
        train(in_dir, out_dir)

    all_demo_npz = []
    for out_dir in offline_feature_dirs:
        for file in tqdm(os.listdir(out_dir), desc=f"Stacking {out_dir}"):
            with np.load(os.path.join(out_dir, file), allow_pickle=True) as offline_data:
                demo_npz = offline_data['data']
            all_demo_npz.append(demo_npz)

    train_data = np.vstack(all_demo_npz)
    print(train_data.shape)
    np.savez(stacked_output_path, data=train_data)

    for in_dir, out_dir in zip(online_tsfel_dirs, online_feature_dirs):
        os.makedirs(out_dir, exist_ok=True)
        train(in_dir, out_dir)



