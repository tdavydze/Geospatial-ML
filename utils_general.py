import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from scipy.io import loadmat
from joblib import load
import yaml
from sklearn.preprocessing import OneHotEncoder
from keras import backend as K
import matplotlib.pyplot as plt
import time
import scipy.io as sio
import gc
from rasterio.transform import from_origin
import rasterio
np.random.seed(42)

# Apply Individual Transformers if transformers_directory is provided
def apply_transformers(dataset = None, transformers_directory = "data/transformers"):
    for feature_name in dataset.columns:
        if feature_name not in {'magnitude', 'other_non_transformed_feature'}:
            transformer_path = os.path.join(transformers_directory, f"{feature_name}_transformer.joblib")
            if os.path.exists(transformer_path):
                transformer = load(transformer_path)
                if isinstance(transformer, OneHotEncoder):
                    transformed_column = transformer.transform(dataset[[feature_name]])
                    transformed_df = pd.DataFrame(
                        # transformed_column.toarray(),
                        transformed_column.toarray().astype(np.uint8),
                        columns=[f"{feature_name}_{cat}" for cat in transformer.categories_[0]],
                        index=dataset.index
                    )
                    dataset.drop(columns=[feature_name], inplace=True)
                    dataset = pd.concat([dataset, transformed_df], axis=1)
                else:
                    # dataset[feature_name] = transformer.transform(dataset[[feature_name]])
                    dataset[feature_name] = transformer.transform(dataset[[feature_name]]).astype(np.float32)
    return dataset

def predict_map(chunks = 3, 
                predict_chunks = [1, 2, 3],
                sub_chunks = 20, 
                transformers_directory = "data/transformers",
                model = None,
                feature_names = None, 
                output_folder = None):
    
    # Load total length and calculate chunk size
    total_length = sio.loadmat(os.path.join(output_folder, "X.mat"))["X"].shape[1]
    chunk_size = total_length // chunks  # Assuming you want to divide the total length by 60 to get the chunk size
    sub_chunk_size = chunk_size // sub_chunks
    
    # Loop over chunks and calculate predictions
    for counter, i in enumerate(range(0, total_length, chunk_size)):
        if (counter + 1) not in predict_chunks:  # Skip the chunks not in predict_chunks
            continue
        entry_start_time = time.time()
        # Load feature data for the current chunk
        df = pd.DataFrame()
        
        # Load feature data for the current chunk directly into the DataFrame
        for feature_index, feature_name in enumerate(feature_names):
            mat_file_path = os.path.join(output_folder, f"{feature_name}.mat")
            df[feature_name] = sio.loadmat(mat_file_path)[feature_name][0,i:i + chunk_size].astype(np.float32)
            print(f"feature {feature_index+1} loaded")
        elapsed_time = time.time() - entry_start_time
        print(f"All features loaded in {elapsed_time} seconds")    
        transformer_start = time.time()
        # Apply transformers and make predictions
        df = apply_transformers(df, transformers_directory)
        transform_time = time.time() - transformer_start
        print(f"All features transformed in {transform_time} seconds")
        try:
            x_data = np.hstack((x_data, sio.loadmat(os.path.join(output_folder, 'X.mat'))['X'][0,i:i + chunk_size]))
            y_data = np.hstack((y_data, sio.loadmat(os.path.join(output_folder, 'Y.mat'))['Y'][0,i:i + chunk_size]))
        except NameError:  # x_data, y_data are not defined
            x_data = sio.loadmat(os.path.join(output_folder, 'X.mat'))['X'][0,i:i + chunk_size]
            y_data = sio.loadmat(os.path.join(output_folder, 'Y.mat'))['Y'][0,i:i + chunk_size]
         
        for k, j in enumerate(range(0, chunk_size, sub_chunk_size)):
            predictions = model.predict(df[j:j + sub_chunk_size])
            # Store the predictions in the corresponding indices of z_data
            try:
                z_data = np.hstack((z_data,predictions.flatten()))
            except NameError:
                z_data = predictions.flatten()
            print(f"Chunk {k+1} processed")
            gc.collect()  # Manually collect garbage
    
    sio.savemat('DATA/Training_Results/predictions.mat', {'predictions': z_data}, do_compression=True)
    del df

    return x_data, y_data, z_data



def create_map(pixel_size = 0.008333, x_data=None, y_data=None, z_data = None, map_path = 'data/training_results/MAP.tif'):
    
    # Determine the extent of your data
    x_min, x_max = x_data.min(), x_data.max()
    y_min, y_max = y_data.min(), y_data.max()
    
    # Calculate the number of pixels in x and y direction
    n_x = int(np.ceil((x_max - x_min) / pixel_size))
    n_y = int(np.ceil((y_max - y_min) / pixel_size))
    
    nodata_value = -9999
    raster = np.full((n_y, n_x), nodata_value, dtype=np.float32)
    
    for x, y, z in zip(x_data, y_data, z_data):
        row = min(max(int((y_max - y) / pixel_size), 0), n_y - 1)
        col = min(max(int((x - x_min) / pixel_size), 0), n_x - 1)
        raster[row, col] = z
    
    transform = from_origin(x_min, y_max, pixel_size, pixel_size)
    
    with rasterio.open(map_path, 'w', driver='GTiff', height=raster.shape[0],
                       width=raster.shape[1], count=1, dtype=str(raster.dtype),
                       crs='EPSG:4326', transform=transform, nodata=nodata_value) as dst:
        dst.write(raster, 1)
    
    



def prepare_dataset(decluster_ratio=1, threshold=None, susceptibility_ratio=1, 
                    transformers_directory='data/transformers', config_path = 'config.yaml'):
    # Initialize the final dataset 
    final_dataset = pd.DataFrame()
    
        
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    feature_directory = config.get('output_folder')
    
    # Iterate over processed .csv files and load them
    processed_directory = 'data/studies/formatted'
    for csv_file in os.listdir(processed_directory):
        if csv_file.endswith('.csv'):
            # Extracting the study_name from the file name using regex.
            study_name = csv_file.replace("processed_", "").rsplit('.', 1)[0]  # Adapted line
            csv_path = os.path.join(processed_directory, csv_file)
            data_df = pd.read_csv(csv_path)
        
            # Decluster the data
            if study_name not in ["study193", "study19221"]:
                data_df = data_df.sample(frac=decluster_ratio) 

            # Apply thresholding if threshold value is provided
            if threshold is not None:
                data_df = data_df[data_df['magnitude'] > threshold]

            final_dataset = pd.concat([final_dataset, data_df], ignore_index=True)
    final_dataset[["X", "Y", "magnitude"]].to_csv('data/training_without_susceptibility.csv', index=False)
    # final_dataset.to_csv('DATA/training.csv', index=False)
    
    # Load Susceptibility_VL.mat and add data to the final dataset
    susceptibility_data = loadmat(os.path.join(feature_directory, 'Susceptibility_VL_data.mat'))['Susceptibility_VL_data']
    num_rows_to_add = int(final_dataset.shape[0] * susceptibility_ratio)
    indices = np.random.choice(susceptibility_data.shape[0], num_rows_to_add, replace=False)
    susceptibility_df = pd.DataFrame(susceptibility_data[indices, :], columns=final_dataset.columns)
    susceptibility_df['magnitude'] = 0
    susceptibility_df.to_csv('data/selected_susceptibility_data.csv', index=False)
    final_dataset = pd.concat([final_dataset, susceptibility_df], ignore_index=True)



    if transformers_directory:
        final_dataset = apply_transformers(final_dataset, transformers_directory)

    # Shuffle the datasets
    final_dataset = shuffle(final_dataset)

    # Reordering columns to make 'magnitude' the last column
    cols = [col for col in final_dataset if col != 'magnitude'] + ['magnitude']
    final_dataset = final_dataset[cols]

    # Save the datasets as .csv
    final_dataset.to_csv('data/training_data_processed.csv', index=False)

    return final_dataset


def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def plot_scat(true, pred, alpha=0.05):
    a = plt.axes(aspect='equal')
    plt.scatter(true, pred, alpha=alpha)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    lims = [min(true)*1.2, max(true)*1.2]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims) 
    plt.show()

def plot_evolution(history, epch):
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.xlim([0, epch])
    plt.ylim([0, min(history.history['loss'])*5])
    plt.show()

import sklearn.metrics, math
def evaluate(true, pred):
    print("\n")
    print("Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(true,pred))
    print("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(true,pred))
    print("Root mean squared error (RMSE): %f" % math.sqrt(sklearn.metrics.mean_squared_error(true,pred)))
    print("R square (R^2):                 %f" % sklearn.metrics.r2_score(true,pred))
    print("R (R):                          %f" % np.sqrt(sklearn.metrics.r2_score(true, pred)))


def save_split(X_train,X_test,y_train,y_test):
    sio.savemat('data/training_results/X_train.mat', {'X_train': X_train})
    sio.savemat('data/training_results/X_test.mat', {'X_test': X_test})
    sio.savemat('data/training_results/y_train.mat', {'y_train': np.expand_dims(y_train,1)})
    sio.savemat('data/training_results/y_test.mat', {'y_test': np.expand_dims(y_test,1)})
    
def load_split():
    X_train = loadmat('data/training_results/X_train.mat')['X_train']
    X_test = loadmat('data/training_results/X_test.mat')['X_test']
    y_train = loadmat('data/training_results/y_train.mat')['y_train']
    y_test = loadmat('data/training_results/y_test.mat')['y_test']
    
    return X_train, X_test, y_train, y_test
    
    
    