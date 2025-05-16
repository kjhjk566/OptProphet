import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

seed = 2048
torch.manual_seed(seed)


class CustomDataset(Dataset):
    """
    A PyTorch Dataset for loading feature-label pairs from NumPy arrays.

    Parameters
    ----------
        features : np.ndarray
            Array of input features.
        labels : np.ndarray
            Array of corresponding labels with shape (N, 2).
    """

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        """
        Returns
        -------
            int
                Total number of samples in the dataset.
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Retrieve one sample and its label as tensors.

        Parameters
        ----------
            idx : int
                Index of the sample to retrieve.

        Returns
        -------
            tuple(torch.FloatTensor, torch.FloatTensor)
                A tuple containing the feature tensor and the 2-element label tensor.
        """
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y1, y2 = self.labels[idx, 0], self.labels[idx, 1]
        return x, torch.tensor([y1, y2], dtype=torch.float32)


class StratifiedSampler(Sampler):
    """
    Sampler that ensures each batch preserves approximate class proportions.

    Parameters
    ----------
        labels : np.ndarray
            Array of shape (N, 2) containing integer labels for two tasks.
        batch_size : int
            Number of samples per batch.
    """

    def __init__(self, labels, batch_size):
        self.labels = np.array(["".join(map(str, label)) for label in labels])
        self.batch_size = batch_size
        unique_labels, label_counts = np.unique(self.labels, return_counts=True)
        self.label_indices = {label: np.where(self.labels == label)[0] for label in unique_labels}
        self.label_proportions = label_counts / len(labels)

    def __iter__(self):
        """
        Generate a shuffled list of indices for one epoch, sampling each class proportionally.
        """
        indices = []
        batch_size_per_label = max(1, self.batch_size // len(self.label_indices))

        for label, index_list in self.label_indices.items():
            n_samples = min(batch_size_per_label, len(index_list))
            sampled_indices = np.random.choice(index_list, n_samples, replace=False)
            indices.extend(sampled_indices)

        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        """
        Returns
        -------
            int
                Total number of samples available.
        """
        return len(self.labels)


def init_weights(m):
    """
    Apply Xavier uniform initialization to Linear layers and zero biases.

    Parameters
    ----------
        m : nn.Module
            The module to initialize.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MMOELayer(nn.Module):
    """
    Multi-gate Mixture-of-Experts (MMoE) layer for multi-task learning.

    Parameters
    ----------
        input_dim : int
            Dimension of the input feature vector.
        num_tasks : int
            Number of prediction tasks.
        num_experts : int
            Number of expert networks.
        expert_dim : int
            Output dimension of each expert.
        seed : int, optional
            Random seed for reproducibility.
    """

    def __init__(self, input_dim, num_tasks, num_experts, expert_dim, seed=1024):
        super(MMOELayer, self).__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.expert_dim = expert_dim

        self.expert_kernel = nn.Linear(input_dim, num_experts * expert_dim, bias=False)
        self.gate_kernels = nn.ModuleList([
            nn.Linear(input_dim, num_experts, bias=False) for _ in range(num_tasks)
        ])

    def forward(self, x):
        """
        Forward pass through the MMoE layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
            list[torch.Tensor]
                A list of length num_tasks, each tensor of shape (batch_size, expert_dim).
        """
        batch_size = x.size(0)
        expert_out = self.expert_kernel(x)
        expert_out = expert_out.view(batch_size, self.expert_dim, self.num_experts)

        task_outputs = []
        for gate_kernel in self.gate_kernels:
            gate_out = F.softmax(gate_kernel(x), dim=-1)
            gate_out = gate_out.unsqueeze(1).repeat(1, self.expert_dim, 1)
            task_out = torch.sum(expert_out * gate_out, dim=2)
            task_outputs.append(task_out)

        return task_outputs


class DNN(nn.Module):
    """
    Deep neural network sequence for feature transformation.

    Parameters
    ----------
        input_dim : int
            Dimension of the input vector.
        hidden_units : tuple[int]
            Sizes of hidden layers.
        activation : str, optional
            Activation function to use ('relu' supported).
        dropout : float, optional
            Dropout probability applied after each layer.
    """

    def __init__(self, input_dim, hidden_units, activation='relu', dropout=0.0):
        super(DNN, self).__init__()
        layers = []
        for i, unit in enumerate(hidden_units):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_units[i - 1], unit))
            if activation == 'relu':
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.dnn = nn.Sequential(*layers)

    def forward(self, x):
        """
        Execute the feedforward pass through the DNN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
            torch.Tensor
                Output tensor after sequential transformations.
        """
        return self.dnn(x)


class PredictionLayer(nn.Module):
    """
    Sigmoid activation layer for binary task predictions.
    """

    def __init__(self):
        super(PredictionLayer, self).__init__()
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """
        Apply sigmoid activation to logits.

        Parameters
        ----------
        x : torch.Tensor
            Logit tensor of arbitrary shape.

        Returns
        -------
            torch.Tensor
                Activated probabilities in the same shape as input.
        """
        return self.activation(x)


class MMOEModel(nn.Module):
    """
    End-to-end MMOE model combining shared and task-specific networks.

    Parameters
    ----------
        input_dim : int
            Dimension of the input feature vector.
        num_tasks : int
            Number of prediction tasks (must be >1).
        num_experts : int, optional
            Number of experts in the MMoE layer.
        expert_dim : int, optional
            Output dimension of each expert.
        dnn_hidden_units : tuple[int], optional
            Hidden units for the shared DNN.
        task_dnn_units : list[int] or None, optional
            Hidden units for each task-specific DNN.
        dropout : float, optional
            Dropout probability.
        seed : int, optional
            Random seed for reproducibility.
    """

    def __init__(self, input_dim, num_tasks, num_experts=4, expert_dim=8,
                 dnn_hidden_units=(128, 128), task_dnn_units=None, dropout=0.0, seed=1024):
        super(MMOEModel, self).__init__()

        if num_tasks <= 1:
            raise ValueError("num_tasks must be greater than 1")

        self.dnn = DNN(input_dim, dnn_hidden_units, activation='relu', dropout=dropout)
        self.mmoe_layer = MMOELayer(input_dim=dnn_hidden_units[-1], num_tasks=num_tasks,
                                    num_experts=num_experts, expert_dim=expert_dim, seed=seed)
        self.task_specific_dnns = nn.ModuleList([
            DNN(expert_dim, task_dnn_units, activation='relu', dropout=dropout) if task_dnn_units else nn.Identity()
            for _ in range(num_tasks)
        ])
        self.prediction_layers = nn.ModuleList([PredictionLayer() for _ in range(num_tasks)])
        self.output_layers = nn.ModuleList([
            nn.Linear(task_dnn_units[-1] if task_dnn_units else expert_dim, 1) for _ in range(num_tasks)
        ])

        self.apply(init_weights)

    def forward(self, x):
        """
        Compute predictions for all tasks.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Concatenated task predictions of shape (batch_size, num_tasks).
        """
        dnn_out = self.dnn(x)
        mmoe_outs = self.mmoe_layer(dnn_out)

        outputs = []
        for i, mmoe_out in enumerate(mmoe_outs):
            task_out = self.task_specific_dnns[i](mmoe_out)
            logit = self.output_layers[i](task_out)
            output = self.prediction_layers[i](logit)
            outputs.append(output)
        return torch.cat(outputs, dim=1)


criterion_task_1 = nn.BCELoss()
criterion_task_2 = nn.BCELoss()


def compute_loss(logits, labels):
    """
    Compute the combined loss for two tasks, gating the second task loss on the first task outcome.

    Parameters
    ----------
    logits : torch.Tensor
        Predicted probabilities of shape (batch_size, 2).
    labels : torch.Tensor
        Ground truth labels of shape (batch_size, 2).

    Returns
    -------
    torch.Tensor
        Scalar tensor representing the total loss.
    """
    task_1_logits, task_2_logits = logits[:, 0], logits[:, 1]
    task_1_labels, task_2_labels = labels[:, 0], labels[:, 1]

    loss_task_1 = criterion_task_1(task_1_logits, task_1_labels)
    valid_task_2 = (task_1_labels > 0.5)

    if valid_task_2.sum() > 0:
        loss_task_2 = criterion_task_2(task_2_logits[valid_task_2], task_2_labels[valid_task_2])
    else:
        loss_task_2 = torch.tensor(0.0, device=logits.device)

    total_loss = loss_task_1 + loss_task_2
    return total_loss


def main_workflow(
    train_data_path,
    model_save_path,
    input_dim,
    num_tasks,
    learning_rate,
    task_dnn_units,
    num_epochs,
    batch_size,
    online_feature_dirs,
    result_dirs
):
    """
    Execute the full training and inference pipeline for the MMOE model.

    Parameters
    ----------
        train_data_path : str
            Path to the training data NPZ file.
        model_save_path : str
            File path where the trained model will be saved ('.pth').
        input_dim : int
            Dimension of the input features.
        num_tasks : int
            Number of tasks for multi-task learning.
        learning_rate : float
            Learning rate for the Adam optimizer.
        task_dnn_units : list[int]
            Hidden layer sizes for each task-specific network.
        num_epochs : int
            Number of training epochs.
        batch_size : int
            Size of each mini-batch.
        online_feature_dirs : list[str]
            List of directories containing NPZ feature files for inference.
        result_dirs : list[str]
            Corresponding output directories for saving CSV predictions.
    """
    model = MMOEModel(input_dim=input_dim, num_tasks=num_tasks, task_dnn_units=task_dnn_units)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    with np.load(train_data_path, allow_pickle=True) as data:
        train_data = data['data']
        train_data[train_data[:, -1] == -1, -1] = 0
        X_train = train_data[:, :-2].astype(np.float32)
        y_train = train_data[:, -2:].astype(np.float32)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    dataloader = DataLoader(torch.utils.data.TensorDataset(X_train, y_train),
                            batch_size=batch_size, shuffle=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in dataloader:
            logits = model(x_batch)
            loss = compute_loss(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
        scheduler.step()

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    for input_dir, output_dir in zip(online_feature_dirs, result_dirs):
        os.makedirs(output_dir, exist_ok=True)
        for file in tqdm(os.listdir(input_dir), desc=f"Processing {input_dir}"):
            with np.load(os.path.join(input_dir, file), allow_pickle=True) as data:
                X_test = data['data'][:, :-2].astype(np.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            with torch.no_grad():
                y_prob = model(X_test)
                y_pred = (y_prob > 0.5).float()
                df_pred = pd.DataFrame(y_pred.numpy(),
                                       columns=['pred_subhealth_label', 'pred_failure_label'])
                df_pred.to_csv(os.path.join(output_dir, file.replace('.npz', '.csv')), index=False)
