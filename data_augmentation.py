import numpy as np
import os
from tqdm import tqdm
from sklearn.neighbors import BallTree
from scipy.stats import median_abs_deviation


def preprocess_data(
    file_path,
    start_feature_idx,
    end_feature_idx,
    save_path
):
    """
    Preprocess data and compute nearest neighbor distances.

    Parameters
    ----------
        file_path : str
            Path to the input NPZ file.
        start_feature_idx : int
            Index of the first feature column to include.
        end_feature_idx : int
            Index of the last feature column to include.
        save_path : str
            Path to save the processed NPZ file.

    Returns
    -------
        np.ndarray: Preprocessed data array including additional columns:
            [nn_distances, nn_indices, type_labels, nn_type_labels, same_type_flags].
    """
    data = np.load(file_path, allow_pickle=True)['data']
    X = data[:, start_feature_idx:end_feature_idx + 1]

    tree = BallTree(X, metric='chebyshev')
    distances, indices = tree.query(X, k=2)
    nn_distances = distances[:, 1]
    nn_indices = indices[:, 1]

    labels = data[:, -2:]
    label0 = labels[:, 0].astype(int)
    label1 = labels[:, 1].astype(int)
    type_labels = np.char.add(label0.astype(str), label1.astype(str)).astype(int)
    nn_type_labels = type_labels[nn_indices]
    same_type_flags = (type_labels == nn_type_labels).astype(int)

    preprocessed_data = np.column_stack((
        data,
        nn_distances,
        nn_indices,
        type_labels,
        nn_type_labels,
        same_type_flags
    ))
    save_path = os.path.join(save_path, "preprocessed_data.npz")
    np.savez(save_path, preprocessed_data=preprocessed_data)

    return preprocessed_data


def epanechnikov_kernel(u):
    """
    Epanechnikov kernel function.

    Parameters
    ----------
        u : array-like
            Input values.

    Returns
    -------
        np.ndarray: Kernel weights for each input.
    """
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)


def analyze_difficulty(
    data,
    save_path
):
    """
    Analyze classification difficulty and compute target sample counts.

    Parameters
    ----------
        data : np.ndarray
            Preprocessed data array. The last five columns must be:
            [nn_distances, nn_indices, type_labels, nn_type_labels, same_type_flags].
        save_path : str
            Path to save the difficulty analysis NPZ file.

    Returns
    -------
        str: Path to the saved NPZ file containing:
            [type_labels, difficult_samples_count, difficulty, target_samples_count].
    """
    nn_distances = data[:, -5]
    type_labels = data[:, -3]
    same_type_flags = data[:, -1]

    unique_labels = np.unique(type_labels)
    bandwidths = {}

    for label in unique_labels:
        type_nn_distances = nn_distances[type_labels == label]
        if type_nn_distances.size == 0:
            bandwidths[label] = 1.0
            continue
        median_distance = np.median(type_nn_distances)
        h_c = max(1.5 * median_distance, 1e-6)
        bandwidths[label] = h_c

    difficulties = []
    difficult_samples_counts = []

    for label in unique_labels:
        mask = (type_labels == label)
        type_nn_distances = nn_distances[mask]
        type_flags = same_type_flags[mask]
        num_c = mask.sum()
        if num_c == 0:
            difficulties.append(0.0)
            difficult_samples_counts.append(0)
            continue

        total = 0.0
        h_c = bandwidths[label]
        for d, flag in zip(type_nn_distances, type_flags):
            if flag == 0:
                u = d / h_c
                total += epanechnikov_kernel(u)
        difficulties.append(total / num_c)
        difficult_samples_counts.append((type_flags == 0).sum())

    difficulties = np.array(difficulties)
    difficult_samples_counts = np.array(difficult_samples_counts)
    mask = difficulties > 1e-5
    unique_labels = unique_labels[mask]
    difficulties = difficulties[mask]
    difficult_samples_counts = difficult_samples_counts[mask]

    if difficulties.size == 0:
        return None

    min_difficulty = difficulties.min()
    max_difficulty = difficulties.max()
    median_difficulty = np.median(difficulties)
    mad_difficulty = median_abs_deviation(difficulties, scale=1.0)
    range_difficulty = max_difficulty - min_difficulty
    n_base = data.shape[0]

    if range_difficulty > 0:
        alpha = 1.0 + (median_difficulty - min_difficulty) / range_difficulty
        beta = 1.0 - mad_difficulty / range_difficulty
    else:
        alpha = beta = 1.0

    difficulty_alpha = np.power(difficulties,alpha)
    sum_difficulty_alpha = difficulty_alpha.sum()
    exp_term = np.exp(-beta * np.abs(difficulties - median_difficulty))
    difficulty_ratios = (difficulty_alpha / sum_difficulty_alpha) * exp_term
    target_samples_counts = (n_base * difficulty_ratios).astype(int)

    save_path = os.path.join(save_path, "analyze_difficulty.npz")
    np.savez(
        save_path,
        type_label=unique_labels,
        difficult_samples_count=difficult_samples_counts,
        difficulty=difficulties,
        target_samples_count=target_samples_counts
    )

    return save_path


def calculate_cluster_iqr(
    data,
    types_to_increase,
    start_feature_idx,
    end_feature_idx,
):
    """
    Compute the interquartile range (IQR) for each feature of specified clusters.

    This function calculates the IQR (Q3 - Q1) for each feature within each target label cluster.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_samples, n_features + metadata) containing all sample data.
    types_to_increase : Sequence[int]
        Iterable of target types for which IQR should be computed.
    start_feature_idx : int
        Index of the first feature column.
    end_feature_idx : int
        Index of the last feature column.

    Returns
    -------
    Dict[int, np.ndarray]
        Mapping from target label to its feature-wise IQR array.
    """
    type_labels_col = -3
    cluster_iqrs = {}

    for target_type in types_to_increase:
        label_indices = np.where(data[:, type_labels_col] == target_type)[0]
        label_data = data[label_indices, start_feature_idx:end_feature_idx + 1]

        q75 = np.percentile(label_data, 75, axis=0)
        q25 = np.percentile(label_data, 25, axis=0)
        iqr = q75 - q25

        cluster_iqrs[target_type] = iqr

    return cluster_iqrs


def update_cluster_iqr(
    cluster_iqrs,
    target_label,
    data,
    start_feature_idx,
    end_feature_idx
):
    """
    Recompute IQR for a specified cluster after adding a new sample.

    This function updates the stored IQR for target_label using all current samples.

    Parameters
    ----------
    cluster_iqrs : Dict[int, np.ndarray]
        Existing mapping from labels to IQR arrays.
    target_label : int
        Label of the cluster to update.
    data : np.ndarray
        Array containing all samples including the new one.
    start_feature_idx : int
        Index of the first feature column.
    end_feature_idx : int
        Index of the last feature column.

    Returns
    -------
    Dict[int, np.ndarray]
        Updated cluster_iqrs with recalculated IQR for target_label.
    """
    label_data = data[data[:, -3] == target_label, start_feature_idx:end_feature_idx + 1]

    q75 = np.percentile(label_data, 75, axis=0)
    q25 = np.percentile(label_data, 25, axis=0)
    iqr = q75 - q25

    cluster_iqrs[target_label] = iqr
    return cluster_iqrs


def feature_weighted_oversampling(
    data,
    target_label,
    existing_count,
    target_count,
    start_feature_idx,
    end_feature_idx,
    cluster_iqrs
):
    """
    Generate synthetic samples to augment a target cluster.

    New samples are created via interpolation between the most difficult samples
    and their nearest same-type neighbors using IQR-based weighting.

    Parameters
    ----------
    data : np.ndarray
        Dataset array, including metadata columns.
    target_label : int
        Label of the cluster to augment.
    existing_count : int
        Current count of samples in the target cluster.
    target_count : int
        Desired total count for the target cluster.
    start_feature_idx : int
        Index of the first feature column.
    end_feature_idx : int
        Index of the last feature column.
    cluster_iqrs : Dict[int, np.ndarray]
        Mapping from each label to its current IQR array.

    Returns
    -------
    tuple
        Updated data array and cluster_iqrs after augmentation.
    """

    nn_distances_col = -5
    nn_indices_col = -4
    type_labels_col = -3
    nn_type_labels_col = -2
    same_type_flags_col = -1

    same_type_data = data[data[:, type_labels_col] == target_label, start_feature_idx:end_feature_idx + 1]
    same_type_indices = np.where(data[:, type_labels_col] == target_label)[0]

    tree = BallTree(same_type_data, metric='chebyshev')
    distances, indices = tree.query(same_type_data, k=2)

    same_type_nn = np.zeros((same_type_data.shape[0], 3))
    same_type_nn[:, 0] = indices[:, 1]
    same_type_nn[:, 1] = distances[:, 1]
    same_type_nn[:, 2] = same_type_indices

    range_features = range(start_feature_idx, end_feature_idx)
    target_iqr = cluster_iqrs[target_label]
    max_iqr = np.max(target_iqr)

    with tqdm(total=target_count - existing_count, desc=f"FWO type {target_label}") as pbar:
        while existing_count < target_count:
            type_indices = np.where(data[:, type_labels_col] == target_label)[0]
            type_D = data[type_indices, nn_distances_col]
            median_D = np.median(type_D)

            psi = type_D / median_D
            idx_in_type = np.argmin(psi)
            seed_idx = type_indices[idx_in_type]
            seed_sample = data[seed_idx]

            new_sample = seed_sample.copy()

            seed_sample_index = np.where(np.all(data == seed_sample, axis=1))[0]
            nn_index_in_same_type = np.where(same_type_nn[:, 2] == seed_sample_index)[0]
            nn_sample_in_same_type = data[nn_index_in_same_type].T

            for feature in range_features:
                iqr_k = target_iqr[feature - start_feature_idx]
                upper_bound = iqr_k / (max_iqr + 1e-6)
                upper_bound = np.asarray(upper_bound, dtype=np.float64)
                lambda_k = np.random.uniform(0, upper_bound)
                new_sample[feature] = (1 - lambda_k) * seed_sample[feature] + lambda_k * nn_sample_in_same_type[feature]

            new_sample[-7:] = seed_sample[-7:]

            data = np.vstack([data, new_sample.T])
            existing_count += 1

            cluster_iqrs = update_cluster_iqr(
                cluster_iqrs=cluster_iqrs,
                target_label=target_label,
                data=data,
                start_feature_idx=start_feature_idx,
                end_feature_idx=end_feature_idx
            )

            target_iqr = cluster_iqrs[target_label]
            max_iqr = np.max(target_iqr)

            new_sample_distances = np.abs(
                data[:, start_feature_idx:end_feature_idx + 1] - new_sample[start_feature_idx:end_feature_idx + 1]
            ).max(axis=1).ravel()
            new_sample_distances[-1] = np.inf

            for idx, distance in enumerate(new_sample_distances[:-1]):
                if distance < data[idx, nn_distances_col]:
                    data[idx, nn_distances_col] = distance
                    data[idx, nn_indices_col] = len(data) - 1
                    if data[idx, type_labels_col] == target_label:
                        neighbor_row = np.where(same_type_nn[:, 2] == idx)[0]
                        same_type_nn[neighbor_row, 0] = len(data) - 1
                        same_type_nn[neighbor_row, 1] = distance
                        same_type_nn[neighbor_row, 2] = idx

            nn_distance_for_new_sample = float(np.min(new_sample_distances))
            nn_index_for_new_sample = np.argmin(new_sample_distances).item()

            data[len(data) - 1, nn_distances_col] = nn_distance_for_new_sample
            data[len(data) - 1, nn_indices_col] = nn_index_for_new_sample
            data[:, nn_indices_col] = data[:, nn_indices_col].astype(int)

            if data[nn_index_for_new_sample, type_labels_col] == target_label:
                new_row = np.array([[nn_index_for_new_sample, nn_distance_for_new_sample, len(data) - 1]])
                same_type_nn = np.vstack([
                    same_type_nn,
                    new_row
                ])
            else:
                same_type_indices = np.where(data[:, type_labels_col] == target_label)[0]
                same_type_distances = new_sample_distances[same_type_indices]
                if same_type_indices.size > 0:
                    nn_distance_for_new_sample = same_type_distances.min().item()
                    nn_index_for_new_sample = same_type_indices[np.argmin(same_type_distances)].item()
                else:
                    nn_distance_for_new_sample = np.inf
                    nn_index_for_new_sample = np.nan

                new_row = np.array([[nn_index_for_new_sample, nn_distance_for_new_sample, len(data) - 1]])
                same_type_nn = np.vstack([
                    same_type_nn,
                    new_row
                ])

            data[:, nn_type_labels_col] = data[data[:, nn_indices_col].astype(int), type_labels_col]
            data[:, same_type_flags_col] = (data[:, type_labels_col] == data[:, nn_type_labels_col]).astype(int)

            pbar.update(1)

    return data, cluster_iqrs


def information_aware_undersampling(
    data,
    target_label,
    existing_count,
    target_count,
    start_feature_idx,
    end_feature_idx
):
    """
    Remove samples from a target cluster to reach a desired count.

    This function deletes the least significant samples based on a PSI metric and
    updates nearest-neighbor references for remaining samples.

    Parameters
    ----------
    data : np.ndarray
        Dataset array including metadata columns.
    target_label : int
        Label of the cluster to reduce.
    existing_count : int
        Current sample count for the target cluster.
    target_count : int
        Desired sample count after reduction.
    start_feature_idx : int
        Index of the first feature column.
    end_feature_idx : int
        Index of the last feature column.

    Returns
    -------
    np.ndarray
        Updated data array after sample removal.
    """
    nn_distances_col = -5
    nearest_indices_col = -4
    type_labels_col = -3
    nearest_true_labels_col = -2
    same_type_flags_col = -1

    with tqdm(total=existing_count - target_count, desc=f"IAU type {target_label}") as pbar:
        while existing_count > target_count:
            type_indices = np.where(data[:, type_labels_col] == target_label)[0]
            type_D = data[type_indices, nn_distances_col]
            median_D = np.median(type_D)

            psi = type_D / median_D
            idx_in_type = np.argmin(psi)
            remove_idx = type_indices[idx_in_type]

            data = np.delete(data, remove_idx, axis=0)
            existing_count -= 1

            data[:, nearest_indices_col] = np.where(
                data[:, nearest_indices_col] > remove_idx,
                data[:, nearest_indices_col] - 1,
                data[:, nearest_indices_col]
            )

            for i in range(len(data)):
                if data[i, nearest_indices_col] == remove_idx:
                    distances = np.abs(
                        data[i, start_feature_idx:end_feature_idx + 1] - data[:, start_feature_idx:end_feature_idx + 1]
                    ).max(axis=1)
                    distances[i] = np.inf
                    min_distance = np.min(distances)
                    nearest_index = np.argmin(distances)

                    data[i, nn_distances_col] = min_distance
                    data[i, nearest_indices_col] = nearest_index
                    data[i, nearest_true_labels_col] = data[nearest_index, type_labels_col]
                    data[i, same_type_flags_col] = (data[i, type_labels_col] == data[i, nearest_true_labels_col])

            pbar.update(1)

    return data


def ensure_directory_exists(directory):
    """
    Ensure the specified directory exists; create it if missing.

    Parameters
    ----------
    directory : str
        Path of the directory to verify or create.

    Returns
    -------
    None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' has been created.")
    else:
        print(f"Directory '{directory}' already exists.")


def process_data(
    train_file,
    start_feature_idx,
    end_feature_idx,
    train_name
):
    """
    Orchestrate full data workflow: preprocessing, difficulty analysis, sampling, and saving.

    Parameters
    ----------
    train_file : str
        Path to the training data file.
    start_feature_idx : int
        Start index of feature columns.
    end_feature_idx : int
        End index of feature columns.
    train_name : str
        Name of the training task for directory naming.

    Returns
    -------
    None
    """
    type_labels_col = -3
    preprocess_data_save_dir = os.path.join('data', train_name, 'preprocess_data')
    analyze_difficulty_save_dir = os.path.join('data', train_name, 'analyze_difficulty')
    result_save_dir = os.path.join('data', train_name, 'result')

    ensure_directory_exists(preprocess_data_save_dir)
    ensure_directory_exists(analyze_difficulty_save_dir)
    ensure_directory_exists(result_save_dir)

    data_file = os.path.join(preprocess_data_save_dir, 'preprocessed_data.npz')
    difficulty_data_file = os.path.join(analyze_difficulty_save_dir, 'analyze_difficulty.npz')

    if os.path.exists(data_file):
        data = np.load(data_file, allow_pickle=True)['preprocessed_data']
        print("Local preprocessed data loaded; skipping preprocessing.")
    else:
        data = preprocess_data(
            file_path=train_file,
            start_feature_idx=start_feature_idx,
            end_feature_idx=end_feature_idx,
            save_path=preprocess_data_save_dir
        )

    if os.path.exists(difficulty_data_file):
        difficulty_data = np.load(difficulty_data_file, allow_pickle=True)
        print("Local difficulty analysis loaded; skipping analysis.")
    else:
        analysis_result = analyze_difficulty(
            data=data,
            save_path=analyze_difficulty_save_dir
        )
        if analysis_result is None:
            print("All types are balanced; no augmentation needed.")
            return
        difficulty_data = np.load(analysis_result, allow_pickle=True)
        print("Difficulty analysis completed.")

    labels_to_adjust = difficulty_data['type_label'].tolist()
    print(f"Labels to adjust: {labels_to_adjust}")

    labels_to_increase = difficulty_data['type_label'][
        difficulty_data['difficult_samples_count'] < difficulty_data['target_samples_count']
    ].tolist()

    cluster_iqrs = calculate_cluster_iqr(
        data=data,
        types_to_increase=labels_to_increase,
        start_feature_idx=start_feature_idx,
        end_feature_idx=end_feature_idx
    )

    for target_label in labels_to_adjust:
        existing_count = np.count_nonzero(data[:, type_labels_col] == target_label)
        target_count = difficulty_data['target_samples_count'][
            difficulty_data['type_label'] == target_label
        ][0]

        print(f"Existing count for label {target_label}: {existing_count}")
        print(f"Target count for label {target_label}: {target_count}")

        if existing_count < target_count:
            print(f"FWO for type {target_label}...")
            data, cluster_iqrs = feature_weighted_oversampling(
                data=data,
                target_label=target_label,
                existing_count=existing_count,
                target_count=target_count,
                start_feature_idx=start_feature_idx,
                end_feature_idx=end_feature_idx,
                cluster_iqrs=cluster_iqrs
            )
        elif existing_count > target_count:
            print(f"IAU for type {target_label}...")
            data = information_aware_undersampling(
                data=data,
                target_label=target_label,
                existing_count=existing_count,
                target_count=target_count,
                start_feature_idx=start_feature_idx,
                end_feature_idx=end_feature_idx
            )
        else:
            print(f"No sampling required for label {target_label}.")

    data = data[:, :-5]
    save_file = os.path.join(result_save_dir, 'results.npz')
    np.savez(save_file, data=data)
    print(f"Augmented data saved to {save_file}.")