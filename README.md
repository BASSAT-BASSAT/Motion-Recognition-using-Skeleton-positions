# Real-Time Shoplifting Detection using Skeleton Poses

This project implements and evaluates an LSTM with an Attention mechanism for real-time shoplifting detection. The model analyzes sequences of 2D skeleton poses extracted from video footage to classify actions as either "Normal" or "Potential Shoplifter".

The repository provides modular code, command-line scripts for training and inference, and a Dockerfile for creating a fully reproducible environment.

---

## Features

-   **Feature Engineering**: Extracts 73 dynamic features per frame, including normalized joint positions, velocities, and body orientation.
-   **LSTM with Attention Model**: Utilizes an LSTM to capture temporal patterns and an Attention mechanism to focus on the most relevant frames for classification.
-   **Class Imbalance Handling**: Implements both standard Cross-Entropy Loss and Focal Loss to compare strategies for handling the imbalanced dataset.
-   **Command-Line Interface**: Includes `main.py` for training and `inference.py` for making predictions on new data.
-   **Docker Support**: A `Dockerfile` is provided for easy, reproducible setup and deployment on any machine with Docker.

---

## Project Structure

```
.
├── csvs_Skeleton_poses_.../  # Raw dataset (must be downloaded, ignored by Git)
├── notebooks/                # Jupyter notebooks for experimentation
├── saved_models/             # Directory for trained model weights (created on run)
├── src/                      # Modular Python source code
│   ├── data_processing.py
│   ├── model.py
│   └── training_utils.py
├── .gitignore
├── Dockerfile                # For building a reproducible Docker container
├── main.py                   # Main script for training the model
├── inference.py              # Script for running inference on a single file
├── README.md                 # This file
└── requirements.txt          # Python package dependencies
```

---

## Setup and Installation

You can set up this project either locally using a virtual environment or with Docker.

### Method 1: Local Setup (Recommended for development)

**Prerequisites:**
-   Python 3.8+
-   Git
-   NVIDIA GPU with CUDA (for training)

**Steps:**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/BASSAT-BASSAT/Motion-Recognition-using-Skeleton-positions.git
    cd Motion-Recognition-using-Skeleton-positions
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Unix/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Dataset:**
     For now the data set is unavailable
### Method 2: Docker Setup (Recommended for reproducibility)

**Prerequisites:**
-   [Docker](https://www.docker.com/get-started)
-   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for GPU support)

**Steps:**

1.  **Clone the repository and download the dataset** as described in steps 1 and 4 of the local setup.

2.  **Build the Docker image:**
    From the project's root directory, run:
    ```bash
    docker build -t shoplifting-detection .
    ```
    This command builds a Docker image named `shoplifting-detection` with all dependencies installed.

---

## Usage (Command Line)

### Training the Model

Use the `main.py` script to train the LSTM model. You can choose between Cross-Entropy (`ce`) and Focal Loss (`focal`).

**Command Structure:**
```bash
python main.py --data-dir <path_to_data> --loss-type <ce_or_focal> --epochs <num_epochs>
```

**Example 1: Train with Cross-Entropy Loss for 10 epochs**
```bash
python main.py --data-dir ./csvs_Skeleton_poses_normal_potential_shoplifter --loss-type ce --epochs 10
```

**Example 2: Train with Focal Loss for 20 epochs**
```bash
python main.py --data-dir ./csvs_Skeleton_poses_normal_potential_shoplifter --loss-type focal --epochs 20
```
This will train the model and save the best weights (`.pth` files) into the `saved_models/` directory.

**Training with Docker:**
To run training inside the container, you need to mount your data and model directories.

```bash
# Make sure you are in the project's root directory
docker run --gpus all -it \
  -v "$(pwd)/csvs_Skeleton_poses_normal_potential_shoplifter:/app/csvs_Skeleton_poses_normal_potential_shoplifter" \
  -v "$(pwd)/saved_models:/app/saved_models" \
  shoplifting-detection \
  python main.py --data-dir /app/csvs_Skeleton_poses_normal_potential_shoplifter --loss-type ce
```

### Running Inference

Use the `inference.py` script to make a prediction on a single, unseen CSV file.

**Command Structure:**
```bash
python inference.py --csv-path <path_to_csv> --model-path <path_to_model.pth> --threshold <value>
```

**Key Arguments:**
-   `--model-path`: Path to the saved model weights (e.g., `saved_models/best_ce_model_realistic.pth`).
-   `--threshold`: The optimal threshold determined during validation. Based on the notebook, a good starting point is `0.59` for the Cross-Entropy model.

**Example: Run inference with the Cross-Entropy model**
```bash
python inference.py \
  --csv-path ./csvs_Skeleton_poses_normal_potential_shoplifter/Potential_shoplifter/video_0001_poses.csv \
  --model-path ./saved_models/best_ce_model_realistic.pth \
  --threshold 0.59
```

**Inference with Docker:**
```bash
docker run --gpus all -it \
  -v "$(pwd)/csvs_Skeleton_poses_normal_potential_shoplifter:/app/csvs_Skeleton_poses_normal_potential_shoplifter" \
  -v "$(pwd)/saved_models:/app/saved_models" \
  shoplifting-detection \
  python inference.py \
    --csv-path /app/csvs_Skeleton_poses_normal_potential_shoplifter/Potential_shoplifter/video_0001_poses.csv \
    --model-path /app/saved_models/best_ce_model_realistic.pth \
    --threshold 0.59
```

---

## License

This project is licensed under the Apcahe License for fully free usage