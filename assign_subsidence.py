import time
import os
import yaml
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.io import loadmat, savemat

def load_config(yaml_file_path):
    with open(yaml_file_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def load_data_points(csv_file):
    df = pd.read_csv(csv_file, header=None, usecols=[0, 1, 2])
    df.columns = ["x", "y", "magnitude"]
    return df

config_path = 'config.yaml'
config = load_config(config_path)
output_folder = config.get('output_folder')

# Load saved X and Y coordinates
X = loadmat(os.path.join(output_folder, 'X.mat'))['X'].flatten()
Y = loadmat(os.path.join(output_folder, 'Y.mat'))['Y'].flatten()

# Create KDTree for fast querying
tree = cKDTree(np.column_stack((X, Y)))

# Load predictor data once outside the loop
feature_names = list(config.get('features', {}).keys())
predictor_data = {
    feature_name: loadmat(os.path.join(output_folder, f"{feature_name}.mat"))[feature_name].flatten()
    for feature_name in feature_names
}

processed_directory = 'data/studies/formatted'

if not os.path.exists(processed_directory):
    os.makedirs(processed_directory)

csv_directory = 'data/studies/original'
for csv_file in os.listdir(csv_directory):
    if csv_file.endswith('.csv'):
        csv_path = os.path.join(csv_directory, csv_file)
        data_df = load_data_points(csv_path)

        final_data = np.zeros((len(data_df), 2 + len(feature_names) + 1))
        
        entry_start_time = time.time()
        for i, (x, y, magnitude) in data_df.iterrows():
            _, closest_index = tree.query([x, y])
            final_data[i, 0] = X[closest_index]
            final_data[i, 1] = Y[closest_index]
            
            print(f"Entry {i} out of {len(data_df)} has been processed for file {csv_file}")
            elapsed_time = time.time() - entry_start_time
            print(f"Entry was assigned in {elapsed_time}")
            
            for j, feature_name in enumerate(feature_names):
                final_data[i, j + 2] = predictor_data[feature_name][closest_index]
            
            final_data[i, -1] = magnitude

        # Convert final data to a DataFrame
        columns = ["x", "y"] + feature_names + ["magnitude"]
        df = pd.DataFrame(final_data, columns=columns)
        
        # Check for duplicate rows based on 'x' and 'y' and compute mean magnitude for duplicates
        df = df.groupby(["x", "y"] + feature_names).agg({'magnitude': 'mean'}).reset_index()

        processed_file_path = os.path.join(processed_directory, f'processed_{csv_file}')
        df.to_csv(processed_file_path, index=False)
        
threshold_distance = 0.02
csv_directory = 'data/studies/original'

# DataFrame to hold all the disregarded points
disregarded_points_df = pd.DataFrame(columns=['X', 'Y', 'magnitude', 'file'])
counter = 0
for csv_file in os.listdir(csv_directory):
    if csv_file.endswith('.csv'):
        csv_path = os.path.join(csv_directory, csv_file)
        data_df = load_data_points(csv_path)

        final_data_list = []
        entry_start_time = time.time()
        
        for i, (x, y, magnitude) in data_df.iterrows():
            distance, closest_index = tree.query([x, y])
            
            # If distance is more than the threshold, add to disregarded_points_df and continue
            if distance > threshold_distance:
                print(f"Entry {i} out of {len(data_df)} skipped due to distance threshold")
                disregarded_points_df = pd.concat([disregarded_points_df, pd.DataFrame({'X': [x], 'Y': [y], 'magnitude': [magnitude], 'file': [csv_file]})], ignore_index=True)
                continue
            
            row_data = [X[closest_index], Y[closest_index]]
            print(f"Entry {i} out of {len(data_df)} has been processed for file {csv_file}")
            
            elapsed_time = time.time() - entry_start_time
            print(f"Entry was assigned in {elapsed_time}")

            for feature_name in feature_names:
                row_data.append(predictor_data[feature_name][closest_index])

            row_data.append(magnitude)
            final_data_list.append(row_data)

        # Convert final_data_list to DataFrame
        final_data_df = pd.DataFrame(final_data_list, columns=['X', 'Y'] + feature_names + ['magnitude'])
        
        # Group by X, Y and calculate the mean of magnitude for duplicates
        final_data_df = final_data_df.groupby(['X', 'Y'] + feature_names, as_index=False).agg({'magnitude': 'mean'})
        counter += np.size(final_data_df,0)

        # Save the processed DataFrame to a new CSV file in another folder
        processed_file_path = os.path.join(processed_directory, f'processed_{csv_file}')
        final_data_df.to_csv(processed_file_path, index=False)

# Save disregarded points to a separate CSV file
disregarded_points_df.to_csv('DATA/disregarded_points.csv', index=False)
print(f"Total disregarded points: {np.size(disregarded_points_df,0)}")
print(f"Total number of new poins after reassignment and averaging: {counter}")