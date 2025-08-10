import numpy as np
import pandas as pd
import json
import os
from glob import glob

def load_and_label_files(data_root: str) -> pd.DataFrame:
    all_csv_files = glob(os.path.join(data_root, '**', '*.csv'), recursive=True)
    labeled_data = [{'path': file_path, 'label': 1 if 'Potential_shoplifter' in file_path else 0} for file_path in all_csv_files]
    df_files = pd.DataFrame(labeled_data)
    print(f"Found {len(df_files)} total CSV files.")
    if not df_files.empty:
        print("\nValue counts for labels:")
        print(df_files['label'].value_counts())
    return df_files

def parse_poses_from_string(poses_str: str) -> np.ndarray:
    try:
        return np.array(json.loads(poses_str)).reshape(18, 2)
    except (json.JSONDecodeError, ValueError):
        return np.zeros((18, 2))

def extract_features(csv_path: str) -> np.ndarray | None:
    df = pd.read_csv(csv_path)
    if df.empty or 'POSES' not in df.columns:
        return None

    raw_poses = np.array(df['POSES'].apply(parse_poses_from_string).tolist())
    
    neck_positions = raw_poses[:, 1:2, :]
    normalized_poses = raw_poses - neck_positions
    
    velocities = np.diff(raw_poses, axis=0, prepend=raw_poses[0:1])
    
    neck_trajectory = raw_poses[:, 1, :]
    deltas = np.diff(neck_trajectory, axis=0, prepend=[neck_trajectory[0]])
    orientation_angles_deg = np.degrees(np.arctan2(deltas[:, 1], deltas[:, 0]))
    
    return np.concatenate([
        normalized_poses.reshape(raw_poses.shape[0], -1),
        velocities.reshape(raw_poses.shape[0], -1),
        orientation_angles_deg.reshape(-1, 1)
    ], axis=1)

def get_group_id(file_path: str) -> str:
    return os.path.basename(file_path).replace('.csv', '')[-23:]

def create_sliding_windows(sequences: list, labels: list, window_size: int, step_size: int):
    windowed_sequences, windowed_labels = [], []
    for i, seq in enumerate(sequences):
        if seq.shape[0] >= window_size:
            for start in range(0, seq.shape[0] - window_size + 1, step_size):
                windowed_sequences.append(seq[start:start + window_size])
                windowed_labels.append(labels[i])
    return np.array(windowed_sequences), np.array(windowed_labels)