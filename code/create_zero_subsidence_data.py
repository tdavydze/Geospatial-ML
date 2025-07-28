import scipy.io as sio
import numpy as np
import os
import yaml

np.random.seed(42)

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
output_folder = config.get('output_folder')

X = sio.loadmat(os.path.join(output_folder,'X.mat'))['X'].flatten()
Y = sio.loadmat(os.path.join(output_folder,'Y.mat'))['Y'].flatten()

susceptibility_data = sio.loadmat(os.path.join(output_folder, 'Susceptibility.mat'))['Susceptibility'].flatten()

feature_names = config.get('features', {}).keys()


feature_data = {}
for feature_name in feature_names:
    mat_file_path = os.path.join(output_folder, f"{feature_name}.mat")
    feature_data[feature_name] = sio.loadmat(mat_file_path)[feature_name].flatten()

indices = np.where(susceptibility_data == 1)[0]

num_indices_to_keep = len(indices) // 10

selected_indices = np.random.choice(indices, size=num_indices_to_keep, replace=False)

x_array = X[selected_indices].reshape(-1, 1)
y_array = Y[selected_indices].reshape(-1, 1)
susceptibility_array = susceptibility_data[selected_indices].reshape(-1, 1)

features_array = np.column_stack([feature_data[feature_name][selected_indices] for feature_name in feature_names])

final_data = np.hstack((x_array, y_array, features_array, susceptibility_array))

sio.savemat(os.path.join(output_folder, 'Susceptibility_VL_data.mat'), {'Susceptibility_VL_data': final_data}, do_compression=True)
