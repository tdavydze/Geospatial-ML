import pandas as pd
import numpy as np
import tensorflow as tf
import utils_general
import train_model
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.io import loadmat, savemat

np.random.seed(42)
tf.random.set_seed(42)

tr = 'data/training_results'
# Load Config (this part is assumed as the function is not provided)
config_path = 'config.yaml'
# Call the function with appropriate parameters (load_split function is assumed)
X_train, X_test, y_train, y_test = utils_general.load_split()

# Load model and weights
model = train_model.model
model.load_weights(f'{tr}/weights.hdf5')

# Make predictions
TEST_Pred = model.predict(X_test)
TRAIN_Pred = model.predict(X_train)

import scipy.io as sio
sio.savemat('data/training_results/test_predictions.mat', {'test_predictions': TEST_Pred})
sio.savemat('data/training_results/train_predictions.mat', {'train_predictions': TRAIN_Pred})

# Evaluate performance
utils_general.plot_scat(y_train, TRAIN_Pred, alpha=0.5)
utils_general.plot_scat(y_test, TEST_Pred, alpha=0.5)
utils_general.evaluate(y_test, TEST_Pred)

_training_data_processed = pd.read_csv('data/training_data_processed.csv')
feature_names = [i for i in _training_data_processed.columns.drop(["X", "Y", "magnitude"])]

if os.path.isfile(f'{tr}/shap_values.mat') and os.path.isfile(f'{tr}/shap_values.mat'):
    shap_values = loadmat(f'{tr}/shap_values.mat')['shap_values']
    shap_random_samples = loadmat(f'{tr}/shap_random_samples.mat')['shap_random_samples']
else:
    # Create SHAP explainer using all features
    n_background_samples = 1000
    indices = np.random.choice(X_train.shape[0], size=n_background_samples, replace=False)
    shap_random_samples = X_train[indices]
    explainer = shap.DeepExplainer(model, shap_random_samples) 
    
    # Compute SHAP values for the whole dataset
    shap_values = explainer.shap_values(X_train)[0]
    savemat(f'{tr}/shap_values.mat', {'shap_values': shap_values})
    savemat(f'{tr}/shap_random_samples.mat', {'shap_random_samples': shap_random_samples})

# When plotting, exclude undesired features
excluded_columns = [i for i in feature_names if i.startswith(("Lithology", "Land_Cover"))]
excluded_indices = [feature_names.index(col) for col in excluded_columns]

included_shap_values = np.delete(shap_values, excluded_indices, axis=1)
included_X_train = np.delete(X_train, excluded_indices, axis=1)
included_feature_names = [i for i in feature_names if not i.startswith(("Lithology", "Land_Cover"))]

# Now plot the SHAP values
n_top_features = 21
shap.summary_plot(included_shap_values, included_X_train, feature_names=included_feature_names, plot_type="bar", max_display=n_top_features)
shap.summary_plot(included_shap_values, included_X_train, plot_type="dot", feature_names=included_feature_names, max_display=n_top_features)

# Calculate the mean absolute SHAP values for the included features
mean_abs_shap_values = np.mean(np.abs(included_shap_values), axis=0)

# Pair the feature names with their respective mean absolute SHAP values
feature_importance = dict(zip(included_feature_names, mean_abs_shap_values))

# Sort the features based on importance
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

# Print the sorted features and their importances
for feature, importance in sorted_features:
    print(f"{feature}: {importance}")
df = pd.DataFrame(sorted_features, columns=['Feature', 'Importance'])

# Save the DataFrame to a .csv file
df.to_csv('data/training_results/feature_importance.csv', index=False)