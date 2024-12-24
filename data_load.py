import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Set input and output directories
input_dir = "/home/ubuntu/Downloads/DroneDetect_V2/CLEAN/"
output_dir = "/home/ubuntu/Downloads/DroneDetect_V2/READY/"
os.makedirs(output_dir, exist_ok=True)

# Define label mapping for device, flight mode, and interference combinations
label_mapping = {
    'AIR_FY': {'AIR_0010': 0}, 'AIR_HO': {'AIR_0001': 1}, 'AIR_ON': {'AIR_0000': 2},
    'DIS_FY': {'DIS_0010': 3}, 'DIS_HO': {'DIS_0001': 4}, 'DIS_ON': {'DIS_0000': 5},
    'INS_FY': {'INS_0010': 6}, 'INS_HO': {'INS_0001': 7}, 'INS_ON': {'INS_0000': 8},
    'MIN_FY': {'MIN_0010': 9}, 'MIN_HO': {'MIN_0001': 10}, 'MIN_ON': {'MIN_0000': 11},
    'MP1_FY': {'MP1_0010': 12}, 'MP1_HO': {'MP1_0001': 13}, 'MP1_ON': {'MP1_0000': 14},
    'MP2_FY': {'MP2_0010': 15}, 'MP2_HO': {'MP2_0001': 16}, 'MP2_ON': {'MP2_0000': 17},
    'PHA_FY': {'PHA_0010': 18}, 'PHA_HO': {'PHA_0001': 19}, 'PHA_ON': {'PHA_0000': 20}
}

# Prepare lists for training and testing tensors
X_train_tensors = []
y_train_tensors = []
X_test_tensors = []
y_test_tensors = []

# Process each folder and file to generate features (X) and labels (y)
for root, dirs, files in os.walk(input_dir):
    for f in files:
        if f.endswith('.h5'):
            file_path = os.path.join(root, f)
            print(f"Processing: {file_path}")

            # Load data from the .h5 file
            df = pd.read_hdf(file_path)

            # Extract device + flight mode from folder name and interference type from file name
            folder_name = os.path.basename(root)  # E.g., "AIR_FY"
            interference_type = f.split('_')[1]  # E.g., "0010" from "AIR_0010_00.h5"
            device_interference = f"{folder_name[:3]}_{interference_type}"  # E.g., "AIR_0010"

            # Look up the label in the label mapping
            label = label_mapping.get(folder_name, {}).get(device_interference)
            if label is None:
                print(f"Skipping file {f} - No matching label in the mapping.")
                continue
            print(f"Label for {folder_name} + {device_interference}: {label}")

            # Process data in chunks to avoid memory issues
            chunk_size = 100000
            total_rows = len(df)

            for start_row in range(0, total_rows, chunk_size):
                end_row = min(start_row + chunk_size, total_rows)
                chunk = df.iloc[start_row:end_row].values.astype(np.float32)

                # Remove rows with NaN values
                chunk = chunk[~np.isnan(chunk).any(axis=1)]
                # Convert to tensor if chunk is not empty
                if len(chunk) > 0:
                    # Convert chunk to tensor
                    chunk_tensor = torch.tensor(chunk, dtype=torch.float32)
                    label_tensor = torch.tensor([label] * len(chunk), dtype=torch.long)

                    # Assign to train or test sets
                    if np.random.rand() < 0.8:
                        X_train_tensors.append(chunk_tensor)
                        y_train_tensors.append(label_tensor)
                    else:
                        X_test_tensors.append(chunk_tensor)
                        y_test_tensors.append(label_tensor)

# Concatenate tensors for train and test sets
if not X_train_tensors or not X_test_tensors:
    raise FileNotFoundError("No valid .h5 files found with matching folder or interference type.")
else:
    X_train_tensor = torch.cat(X_train_tensors, dim=0)
    y_train_tensor = torch.cat(y_train_tensors, dim=0)
    X_test_tensor = torch.cat(X_test_tensors, dim=0)
    y_test_tensor = torch.cat(y_test_tensors, dim=0)

print(f"Train set shapes: X_train={X_train_tensor.shape}, y_train={y_train_tensor.shape}")
print(f"Test set shapes: X_test={X_test_tensor.shape}, y_test={y_test_tensor.shape}")

# Save tensors to the output directory
torch.save(X_train_tensor, os.path.join(output_dir, "X_train.pt"))
torch.save(X_test_tensor, os.path.join(output_dir, "X_test.pt"))
torch.save(y_train_tensor, os.path.join(output_dir, "y_train.pt"))
torch.save(y_test_tensor, os.path.join(output_dir, "y_test.pt"))

# Create DataLoader for training and testing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Print summary
print(f"Number of classes: {len(torch.unique(y_train_tensor))}")
