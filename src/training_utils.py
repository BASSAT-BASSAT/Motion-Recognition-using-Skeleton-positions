import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve
from src.data_preprocessing import parse_poses_from_string

def run_online_training_trial(
    model,
    train_loader,
    val_windowed_loader,
    val_long_sequences,
    val_long_labels,
    criterion,
    optimizer,
    num_epochs,
    model_save_path_window,
    model_save_path_realistic,
    device,
    WINDOW_SIZE_FRAMES,
    STEP_SIZE_FRAMES
):
    best_val_acc_window = 0.0
    best_val_acc_realistic = 0.0
    history = {'train_loss': [], 'val_acc_window': [], 'val_acc_realistic': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} | Training")
        for sequences, labels in train_pbar:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        correct_window, total_window = 0, 0
        with torch.no_grad():
            for sequences, labels in val_windowed_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                total_window += labels.size(0)
                correct_window += (predicted == labels).sum().item()
        epoch_acc_val_window = 100 * correct_window / total_window
        history['val_acc_window'].append(epoch_acc_val_window)

        epoch_acc_val_realistic = evaluate_realistically(
            model, val_long_sequences, val_long_labels, device, WINDOW_SIZE_FRAMES, STEP_SIZE_FRAMES
        )
        history['val_acc_realistic'].append(epoch_acc_val_realistic)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Window Val Acc: {epoch_acc_val_window:.2f}% | "
              f"Realistic Val Acc: {epoch_acc_val_realistic:.2f}%")
        
        if epoch_acc_val_window > best_val_acc_window:
            best_val_acc_window = epoch_acc_val_window
            torch.save(model.state_dict(), model_save_path_window)
            print(f"---> New best WINDOW model saved (Val Acc: {best_val_acc_window:.2f}%)")
            
        if epoch_acc_val_realistic > best_val_acc_realistic:
            best_val_acc_realistic = epoch_acc_val_realistic
            torch.save(model.state_dict(), model_save_path_realistic)
            print(f"---> New best REALISTIC model saved (Val Acc: {best_val_acc_realistic:.2f}%)")
            
    print("\n--- Finished Training Trial ---")
    return history

def evaluate_and_report_on_test_set(
    model,
    test_sequences,
    test_labels,
    device,
    WINDOW_SIZE_FRAMES,
    STEP_SIZE_FRAMES
):
    model.eval()
    all_window_preds, all_window_labels = [], []
    video_suspicion_scores = []

    with torch.no_grad():
        for i, long_seq_np in enumerate(tqdm(test_sequences, desc="Evaluating Test Set")):
            y_true_video = test_labels[i]
            seq_windows_tensors = []
            
            if long_seq_np.shape[0] >= WINDOW_SIZE_FRAMES:
                for start in range(0, long_seq_np.shape[0] - WINDOW_SIZE_FRAMES + 1, STEP_SIZE_FRAMES):
                    window = long_seq_np[start:start + WINDOW_SIZE_FRAMES]
                    seq_windows_tensors.append(torch.from_numpy(window))
            
            if not seq_windows_tensors:
                video_suspicion_scores.append(0.0)
                continue

            windows_batch = torch.stack(seq_windows_tensors).float().to(device)
            outputs = model(windows_batch)
            
            _, predicted_windows = torch.max(outputs, 1)
            all_window_preds.extend(predicted_windows.cpu().numpy())
            all_window_labels.extend([y_true_video] * len(predicted_windows))
            
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            avg_suspicion_score = probabilities.mean().item()
            video_suspicion_scores.append(avg_suspicion_score)

    plot_report_and_matrix(all_window_labels, all_window_preds, "Test Set Evaluation (Per-Window)")
    
    simple_accuracy = evaluate_realistically(model, test_sequences, test_labels, device, WINDOW_SIZE_FRAMES, STEP_SIZE_FRAMES)
    print(f"\n--- Test Set Evaluation (Simple Per-Video) ---")
    print(f"Accuracy (if any window is 1, predict 1): {simple_accuracy:.2f}%")

    y_true_video_labels = test_labels
    precisions, recalls, thresholds = precision_recall_curve(y_true_video_labels, video_suspicion_scores)
    f1_scores = np.nan_to_num(2 * (precisions * recalls) / (precisions + recalls))
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    y_pred_thresholded = [1 if score >= best_threshold else 0 for score in video_suspicion_scores]
    plot_report_and_matrix(y_true_video_labels, y_pred_thresholded, f"Test Set Evaluation (Optimal Threshold: {best_threshold:.2f})")

def plot_report_and_matrix(y_true, y_pred, title):
    print(f"\n--- {title} ---")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Potential Shoplifter'], zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Normal', 'Potential Shoplifter'], yticklabels=['Normal', 'Potential Shoplifter'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

def evaluate_realistically(model, long_sequences, long_labels, device, window_size, step_size):
    model.eval()
    y_pred_final = []
    with torch.no_grad():
        for long_seq_np in long_sequences:
            seq_windows_tensors = []
            if long_seq_np.shape[0] >= window_size:
                for start in range(0, long_seq_np.shape[0] - window_size + 1, step_size):
                    seq_windows_tensors.append(torch.from_numpy(long_seq_np[start:start + window_size]))
            
            if not seq_windows_tensors:
                final_video_prediction = 0
            else:
                outputs = model(torch.stack(seq_windows_tensors).float().to(device))
                final_video_prediction = 1 if 1 in torch.max(outputs, 1)[1] else 0
            y_pred_final.append(final_video_prediction)
    return accuracy_score(long_labels, y_pred_final) * 100

def plot_history(history: dict, trial_name: str):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color='tab:red')
    ax1.plot(history['train_loss'], 'r-', label='Training Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Accuracy (%)', color='tab:blue')
    ax2.plot(history['val_acc_window'], 'b--', label='Window Val Acc')
    ax2.plot(history['val_acc_realistic'], 'g-.', label='Realistic Val Acc')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    fig.tight_layout()
    fig.suptitle(f'{trial_name} - Training History', y=1.03)
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    plt.show()

def plot_trajectory(csv_path: str, ax: plt.Axes):
    df = pd.read_csv(csv_path)
    if df.empty or 'POSES' not in df.columns:
        return
    raw_poses = np.array(df['POSES'].apply(parse_poses_from_string).tolist())
    neck_trajectory = raw_poses[:, 1, :]
    neck_trajectory = neck_trajectory[np.any(neck_trajectory != 0, axis=1)]
    
    if neck_trajectory.shape[0] > 1:
        ax.plot(neck_trajectory[:, 0], neck_trajectory[:, 1], label='Trajectory')
        ax.set_xlabel('x (in pixels)')
        ax.set_ylabel('y (in pixels)')
        ax.legend(fontsize='small')
        ax.invert_yaxis()