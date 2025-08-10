import torch
import numpy as np
import argparse
import os

from src.data_preprocessing import extract_features
from src.model import RealTimeClassifier

def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found at {args.csv_path}")
        return
        
    print(f"Loading model from {args.model_path}...")
    model = RealTimeClassifier(
        input_size=73, hidden_size=128, num_layers=2, num_classes=2
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    print(f"Loading and extracting features from {args.csv_path}...")
    features = extract_features(args.csv_path)
    if features is None or features.shape[0] < args.window_size:
        print(f"Not enough frames for a full window. Found {features.shape[0] if features is not None else 0} frames.")
        return

    seq_windows_tensors = []
    for start in range(0, features.shape[0] - args.window_size + 1, args.step_size):
        window = features[start : start + args.window_size]
        seq_windows_tensors.append(torch.from_numpy(window))

    if not seq_windows_tensors:
        print("Could not create any windows from the provided CSV.")
        return

    windows_batch = torch.stack(seq_windows_tensors).float().to(device)
    print(f"Created {len(windows_batch)} windows for inference.")

    with torch.no_grad():
        outputs = model(windows_batch)
        probabilities = torch.softmax(outputs, dim=1)
        
        avg_suspicion_score = probabilities[:, 1].mean().item()
        
        _, predicted_windows = torch.max(outputs, 1)

    print("\n--- Inference Results ---")
    print(f"Average Suspicion Score (Probability of Shoplifter): {avg_suspicion_score:.4f}")

    final_prediction = "Potential Shoplifter" if avg_suspicion_score >= args.threshold else "Normal"
    print(f"Final Prediction (Threshold={args.threshold}): {final_prediction}")

    print("\n--- Per-Window Details ---")
    normal_windows = (predicted_windows == 0).sum().item()
    shoplifter_windows = (predicted_windows == 1).sum().item()
    print(f"Windows classified as 'Normal': {normal_windows}")
    print(f"Windows classified as 'Potential Shoplifter': {shoplifter_windows}")
    print("--------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on a single CSV file for Shoplifting Detection.")
    parser.add_argument('--csv-path', type=str, required=True, help="Path to the input CSV file with skeleton poses.")
    parser.add_argument('--model-path', type=str, required=True, help="Path to the trained .pth model file (e.g., 'saved_models/best_ce_model_realistic.pth').")
    parser.add_argument('--threshold', type=float, default=0.59, help="Suspicion score threshold for classification. Found during validation (e.g., 0.59 for CE, 0.76 for Focal).")
    parser.add_argument('--window-size', type=int, default=150, help="Size of the sliding window in frames.")
    parser.add_argument('--step-size', type=int, default=30, help="Step of the sliding window in frames.")
    
    args = parser.parse_args()
    infer(args)