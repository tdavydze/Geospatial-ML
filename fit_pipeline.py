import os
import yaml
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, StandardScaler, OneHotEncoder, MinMaxScaler
from joblib import dump
from scipy.io import loadmat
from sklearn.pipeline import Pipeline

# Load Config
config_path = 'config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

feature_directory = config.get('output_folder')
transformers_directory = 'data/transformers'  # Directory to store individual transformers

# Ensure the transformers directory exists
if not os.path.exists(transformers_directory):
    os.makedirs(transformers_directory)

# Fit and save individual transformers for each feature
for feature_name, properties in config['features'].items():
    if properties['Transform']:
        # Load feature data
        feature_mat = loadmat(os.path.join(feature_directory, f"{feature_name}.mat"))
        feature_data = pd.DataFrame(feature_mat[feature_name].flatten(), columns=[feature_name])
        
        # Define and fit transformer
        transformer = OneHotEncoder() if properties['OneHot'] else Pipeline([
        ('standard', StandardScaler()),
        ('quantile', QuantileTransformer(output_distribution='normal')),
        ('minmax', MinMaxScaler())
        ])
        
        transformer.fit(feature_data)
        
        # Save the fitted transformer
        dump(transformer, os.path.join(transformers_directory, f"{feature_name}_transformer.joblib"))

print("Transformers have been fitted and saved.")
