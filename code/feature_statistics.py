import os
import numpy as np
import pandas as pd
import scipy.io as sio
import yaml

# Load the config file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

output_folder = config.get('output_folder')
feature_names = list(config.get('features', {}).keys())

# Create an empty dictionary to store statistics
stats_dict = {}

# Loop through each feature name and process the corresponding .mat file
for feature in feature_names:
    mat_filepath = os.path.join(output_folder, f"{feature}.mat")
    if os.path.exists(mat_filepath):
        # Load the .mat file data
        data = sio.loadmat(mat_filepath)

        # Assuming the data is stored in a key named 'data' in the .mat file
        # Modify if the key is different
        array_data = data[f"{feature}"]

        # Compute the statistics and store in the dictionary
        stats_dict[feature] = {
            'Min': np.min(array_data),
            'Max': np.max(array_data),
            'Mean': np.mean(array_data),
            'STD': np.std(array_data)
        }

# Load the feature_importance.csv file
df_importance = pd.read_csv('data/training_results/feature_importance.csv')

# Merge the statistics with the feature importance DataFrame
for index, row in df_importance.iterrows():
    feature = row['Feature']
    if feature in stats_dict:
        for stat, value in stats_dict[feature].items():
            df_importance.at[index, stat] = value

# Save the updated DataFrame
df_importance.to_csv('data/training_results/feature_statistics.csv', index=False)
