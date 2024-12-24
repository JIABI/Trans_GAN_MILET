import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths for data
data_dir = "/home/ubuntu/Downloads/DroneDetect_V2/READY"
h5_dir = "/home/ubuntu/Downloads/DroneDetect_V2/CLEAN/"

# Define label mapping (adjust based on your actual folder structure)
label_mapping = {
    "AIR_FY": 0,
    "AIR_HO": 1,
    "AIR_ON": 2,
    "DIS_FY": 3,
    "DIS_ON": 4,
    "INS_FY": 5,
    "INS_HO": 6,
    "INS_ON": 7,
    "MIN_FY": 8,
    "MIN_HO": 9,
    "MIN_ON": 10,
    "MP1_FY": 11,
    "MP1_HO": 12,
    "MP1_ON": 13,
    "MP2_FY": 14,
    "MP2_HO": 15,
    "MP2_ON": 16,
    "PHA_FY": 17,
    "PHA_HO": 18,
    "PHA_ON": 19
}

# Load data from .h5 files and convert to PyTorch tensors
X_list = []
y_list = []

for root, dirs, files in os.walk(h5_dir):
    for file in files:
        if file.endswith(".h5"):
            file_path = os.path.join(root, file)
            print(f"Processing file: {file_path}")
            folder_name = os.path.basename(root)
            # Check if folder name exists in the label mapping
            if folder_name not in label_mapping:
                print(f"Warning: Folder name '{folder_name}' not found in label mapping. Skipping file.")
                continue
            label = label_mapping[folder_name]
            try:
                # Check available keys in the .h5 file
                with pd.HDFStore(file_path, 'r') as store:
                    print(f"Available keys in {file_path}: {store.keys()}")
                    # Update the key as per the actual structure of your HDF5 file, e.g., '/data' or any other available key
                    df = pd.read_hdf(store, key='/data')  # Replace '/data' with the correct key if necessary
                    if df.shape[0]!=600000:
                        os.remove(file_path)
                        continue
            except KeyError as e:
                print(f"KeyError: {e}. Please verify the key in the HDF5 file.")
                continue
            # Assuming the label is encoded based on folder name
            data = df.to_numpy()
            labels = np.full((data.shape[0],), label)  # Use the label extracted from folder name
            X_list.append(torch.tensor(data, dtype=torch.float32))
            y_list.append(torch.tensor(labels, dtype=torch.long))

# Concatenate all data
if X_list and y_list:  # Ensure lists are not empty before concatenation
    X_tensor = torch.cat(X_list, dim=0)
    y_tensor = torch.cat(y_list, dim=0)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_tensor.numpy())

    # Apply PCA to reduce dimensions
    n_pca_components = 32
    pca = PCA(n_components=n_pca_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Apply NCA to further reduce dimensions
    n_nca_components = 16
    nca = NeighborhoodComponentsAnalysis(n_components=n_nca_components, random_state=42)
    X_nca = nca.fit_transform(X_pca, y_tensor.numpy())

    # Convert back to tensor
    X_tensor_nca = torch.tensor(X_nca, dtype=torch.float32)

    # Split data into train and test sets
    train_ratio = 0.8
    num_samples = X_tensor_nca.shape[0]
    num_train = int(train_ratio * num_samples)

    X_train_tensor = X_tensor_nca[:num_train]
    y_train_tensor = y_tensor[:num_train]
    X_test_tensor = X_tensor_nca[num_train:]
    y_test_tensor = y_tensor[num_train:]

    # Save train and test datasets
    torch.save(X_train_tensor, f"{data_dir}/X_train.pt")
    torch.save(y_train_tensor, f"{data_dir}/y_train.pt")
    torch.save(X_test_tensor, f"{data_dir}/X_test.pt")
    torch.save(y_test_tensor, f"{data_dir}/y_test.pt")
else:
    print("No valid data found to process.")