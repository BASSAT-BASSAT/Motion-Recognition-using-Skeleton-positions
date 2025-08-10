import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import os
import random
import argparse

from src.data_preprocessing import (
    load_and_label_files,
    extract_features,
    get_group_id,
    create_sliding_windows
)
from src.model import RealTimeClassifier, FocalLoss
from src.training_utils import (
    run_online_training_trial,
    evaluate_and_report_on_test_set,
    plot_history
)

def set_seed(seed_value: int):
    """Sets the seed for reproducibility for all relevant libraries."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Global seed set to {seed_value}")

def main(args):
    # Configuration & Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Models will be saved in: {args.save_dir}")

    # Data Loading and Processing
    df_files = load_and_label_files(args.data_dir)

    print("--- Processing all CSV files into feature sequences ---")
    all_sequences, all_labels, all_groups = [], [], []
    for _, row in df_files.iterrows():
        features = extract_features(row['path'])
        if features is not None and len(features) > 0:
            all_sequences.append(features)
            all_labels.append(row['label'])
            all_groups.append(get_group_id(row['path']))

    print("\n--- Splitting data by group to prevent leakage ---")
    gss_train_temp = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_indices, temp_indices = next(gss_train_temp.split(all_sequences, all_labels, all_groups))

    train_sequences = [all_sequences[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    temp_sequences = [all_sequences[i] for i in temp_indices]
    temp_labels = [all_labels[i] for i in temp_indices]
    temp_groups = np.array(all_groups)[temp_indices]
    
    gss_val_test = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=args.seed)
    val_group_indices, test_group_indices = next(gss_val_test.split(temp_sequences, temp_labels, temp_groups))
    
    val_indices = [temp_indices[i] for i in val_group_indices]
    test_indices = [temp_indices[i] for i in test_group_indices]
    
    X_val_long = [all_sequences[i] for i in val_indices]
    y_val_long = [all_labels[i] for i in val_indices]
    X_test_long = [all_sequences[i] for i in test_indices]
    y_test_long = [all_labels[i] for i in test_indices]

    # Create Datasets and DataLoaders
    X_train_windowed, y_train_windowed = create_sliding_windows(
        train_sequences, train_labels, args.window_size, args.step_size
    )
    X_val_windowed, y_val_windowed = create_sliding_windows(
        X_val_long, y_val_long, args.window_size, args.step_size
    )

    train_dataset = TensorDataset(torch.from_numpy(X_train_windowed).float(), torch.from_numpy(y_train_windowed).long())
    val_windowed_dataset = TensorDataset(torch.from_numpy(X_val_windowed).float(), torch.from_numpy(y_val_windowed).long())
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_windowed_loader = DataLoader(val_windowed_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("\n--- Data Preparation Complete ---")
    print(f"Training on {len(X_train_windowed)} windows from {len(train_sequences)} videos.")
    print(f"Validating on {len(X_val_long)} full-length videos.")
    print(f"Testing on {len(X_test_long)} full-length videos.")

    # Model, Loss, and Optimizer Initialization
    model = RealTimeClassifier(
        input_size=73, hidden_size=128, num_layers=2, num_classes=2
    ).to(device)

    model_prefix = ""
    if args.loss_type == 'ce':
        print("\n--- Using Cross-Entropy Loss ---")
        criterion = nn.CrossEntropyLoss()
        model_prefix = "ce"
    elif args.loss_type == 'focal':
        print("\n--- Using Focal Loss ---")
        labels_array = np.array(all_labels)
        class_counts = np.bincount(labels_array)
        alpha_class_0 = class_counts[1] / len(labels_array)
        alpha_class_1 = class_counts[0] / len(labels_array)
        alpha_tensor = torch.tensor([alpha_class_0, alpha_class_1]).to(device)
        criterion = FocalLoss(alpha=alpha_tensor, gamma=2.0)
        model_prefix = "focal"

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    model_path_window = os.path.join(args.save_dir, f'best_{model_prefix}_model_window.pth')
    model_path_realistic = os.path.join(args.save_dir, f'best_{model_prefix}_model_realistic.pth')

    # Training
    history = run_online_training_trial(
        model=model,
        train_loader=train_loader,
        val_windowed_loader=val_windowed_loader,
        val_long_sequences=X_val_long,
        val_long_labels=y_val_long,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        model_save_path_window=model_path_window,
        model_save_path_realistic=model_path_realistic,
        device=device,
        WINDOW_SIZE_FRAMES=args.window_size,
        STEP_SIZE_FRAMES=args.step_size
    )

    # Final Evaluation
    print("\n--- Loading best model for final evaluation on the Test Set ---")
    plot_history(history, f"{args.loss_type.upper()} Loss")
    
    final_model = RealTimeClassifier(input_size=73, hidden_size=128, num_layers=2, num_classes=2).to(device)
    final_model.load_state_dict(torch.load(model_path_realistic))
    print(f"Model loaded successfully from: {model_path_realistic}")

    evaluate_and_report_on_test_set(
        model=final_model,
        test_sequences=X_test_long,
        test_labels=y_test_long,
        device=device,
        WINDOW_SIZE_FRAMES=args.window_size,
        STEP_SIZE_FRAMES=args.step_size
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train LSTM with Attention for Shoplifting Detection.")
    parser.add_argument('--data-dir', type=str, required=True, help="Path to the root directory of the CSV dataset.")
    parser.add_argument('--loss-type', type=str, default='ce', choices=['ce', 'focal'], help="Loss function to use: 'ce' for Cross-Entropy or 'focal' for Focal Loss.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--batch-size', type=int, default=128, help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--save-dir', type=str, default='saved_models', help="Directory to save trained models.")
    parser.add_argument('--window-size', type=int, default=150, help="Size of the sliding window in frames.")
    parser.add_argument('--step-size', type=int, default=30, help="Step of the sliding window in frames.")
    
    args = parser.parse_args()
    main(args)