import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tsfel_feature_extractor import process_tsfel_data
from transformer import full_transformer_workflow
from data_augmentation import process_data
from moe import main_workflow

def main():
    """
    Run the complete machine learning pipeline consisting of the following stages:
    1. TSFEL feature extraction from raw CSV time-series data.
    2. Transformer-based temporal feature modeling and stacking.
    3. Sample-level data augmentation based on feature difficulty.
    4. Multi-task learning using MMOE for final classification and inference.

    Command-line arguments control window size, data paths, feature range, and training settings.
    """
    parser = argparse.ArgumentParser(description="Complete ML Pipeline: TSFEL -> Transformer -> Data Augmentation -> MMOE")

    # === Basic global parameters ===
    parser.add_argument('--window_size', type=int, default=24, help='Sliding window size for TSFEL feature extraction.')
    parser.add_argument('--step_size', type=int, default=1, help='Step size for TSFEL sliding window.')
    parser.add_argument('--first_feature', type=int, default=0, help='Index of the first feature column for augmentation.')
    parser.add_argument('--last_feature', type=int, default=2311, help='Index of the last feature column for augmentation.')
    parser.add_argument('--train_name', type=str, default='train_task', help='Identifier name for the augmentation task.')

    # === Directory arguments (with default values) ===
    parser.add_argument('--offline_input_dirs', nargs='+', default=['./data/offline/hard/', './data/offline/soft/', './data/offline/normal/'], help='List of directories containing offline input CSVs.')
    parser.add_argument('--offline_output_dirs', nargs='+', default=['./data/offline_tsfel/hard/', './data/offline_tsfel/soft/', './data/offline_tsfel/normal/'], help='List of directories to save extracted offline TSFEL features.')
    parser.add_argument('--online_input_dirs', nargs='+', default=['./data/online/hard/', './data/online/soft/', './data/online/normal/'], help='List of directories containing online input CSVs.')
    parser.add_argument('--online_output_dirs', nargs='+', default=['./data/online_tsfel/hard/', './data/online_tsfel/soft/', './data/online_tsfel/normal/'], help='List of directories to save extracted online TSFEL features.')

    parser.add_argument('--offline_feature_dirs', nargs='+', default=['./data/offline_feature/hard/', './data/offline_feature/soft/', './data/offline_feature/normal/'], help='List of directories to save offline Transformer outputs.')
    parser.add_argument('--online_feature_dirs', nargs='+', default=['./data/online_feature/hard/', './data/online_feature/soft/', './data/online_feature/normal/'], help='List of directories to save online Transformer outputs.')

    parser.add_argument('--stacked_output_path', type=str, default='./data/feature_two_step_train.npz', help='Path to save the stacked offline features.')

    parser.add_argument('--result_dirs', nargs='+', default=['./result/hard/', './result/soft/', './result/normal/'], help='List of directories to save final inference results.')

    args = parser.parse_args()

    # ===== Step 1: TSFEL feature extraction =====
    metric_used = [
        'Current', 'Temperature', 'Voltage', 'CurrentTXPower', 'CurrentRXPower',
        'CurrentMultiTXPower1', 'CurrentMultiTXPower2', 'CurrentMultiTXPower3', 'CurrentMultiTXPower4',
        'CurrentMultiRXPower1', 'CurrentMultiRXPower2', 'CurrentMultiRXPower3', 'CurrentMultiRXPower4',
        'CurrentMultiBias1', 'CurrentMultiBias2', 'CurrentMultiBias3', 'CurrentMultiBias4'
    ]

    process_tsfel_data(metric_used, args.window_size, args.step_size, args.offline_input_dirs, args.offline_output_dirs, 'offline')
    process_tsfel_data(metric_used, args.window_size, args.step_size, args.online_input_dirs, args.online_output_dirs, 'online')

    # ===== Step 2: Transformer-based temporal modeling and feature stacking =====
    full_transformer_workflow(
        offline_tsfel_dirs=args.offline_output_dirs,
        offline_feature_dirs=args.offline_feature_dirs,
        online_tsfel_dirs=args.online_output_dirs,
        online_feature_dirs=args.online_feature_dirs,
        stacked_output_path=args.stacked_output_path
    )

    # ===== Step 3: Data augmentation based on sample difficulty =====
    process_data(
        train_file=args.stacked_output_path,
        start_feature_idx=args.first_feature,
        end_feature_idx=args.last_feature,
        train_name=args.train_name
    )

    # ===== Step 4: Multi-task learning using MMOE =====
    main_workflow(
        train_data_path=f'./data/{args.train_name}/result/results.npz',
        model_save_path='./model.pth',
        input_dim=2346,
        num_tasks=2,
        learning_rate=0.001,
        task_dnn_units=[512, 1024, 512],
        num_epochs=30,
        batch_size=2**16,
        online_feature_dirs=args.online_feature_dirs,
        result_dirs=args.result_dirs
    )

if __name__ == "__main__":
    main()
