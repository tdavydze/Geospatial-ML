import os
import numpy as np
from scipy.io import loadmat, savemat
import yaml

# Load Configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
directory = config.get('output_folder')
feature_files = [f"{feature}.mat" for feature in config['features'].keys()]
other_files = [f"{file}.mat" for file in config['other_maps'].keys()]
indices_file_path = os.path.join(directory, "no_data_indices.npy")

# Step 1: Find No-Data Indices
if os.path.exists(indices_file_path):
    no_data_indices = np.load(indices_file_path).tolist()
    print(f"No-data indices loaded from {indices_file_path}")
else:
    no_data_indices = set()
    for file in feature_files:
        file_path = os.path.join(directory, file)
        data_key = os.path.splitext(file)[0]
        
        if os.path.exists(file_path):
            data = loadmat(file_path)[data_key]
            no_data_count_before = np.sum(np.isnan(data))
            print(f"{file}: No-data count before removal: {no_data_count_before}")
            
            indices = set(np.where(np.isnan(data))[1])  # Assuming data is 2D
            no_data_indices = no_data_indices.union(indices)
        else:
            print(f"{file} does not exist. Skipping...")
    # Save no-data indices to a file
    np.save(indices_file_path, np.array(list(no_data_indices)))
    print(f"No-data indices saved to {indices_file_path}")

total_rows = len(no_data_indices)
print(f"Total number of rows to be removed: {total_rows}")

# Step 2: Remove No-Data Indices and Save Cleaned Files
all_files = feature_files + other_files + ["X.mat", "Y.mat"]

for file in all_files:
    file_path = os.path.join(directory, file)
    data_key = os.path.splitext(file)[0]
    
    if os.path.exists(file_path):
        data = loadmat(file_path)[data_key]
        
        # Check if the file has already been cleaned
        if data.size <= (195936292 - total_rows):
            print(f"{file} has already been cleaned. Skipping...")
            continue
        
        cleaned_data = np.delete(data, list(no_data_indices), axis=1)  # Assuming data is 2D
        
        no_data_count_after = np.sum(np.isnan(cleaned_data))
        print(f"{file}: No-data count after removal: {no_data_count_after}")
        
        savemat(file_path, {data_key: cleaned_data}, do_compression=True)
    else:
        print(f"{file} does not exist. Skipping...")
