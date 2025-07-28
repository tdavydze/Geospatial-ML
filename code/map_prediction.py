import utils_general
import train_model
import yaml


model = train_model.model
model.load_weights('data/training_results/weights.hdf5')

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
output_folder = config.get('output_folder')
feature_names = config.get('features', {}).keys()


x_data, y_data, z_data = utils_general.predict_map(
                chunks = 3,
                predict_chunks = [1, 2, 3],
                sub_chunks = 20, 
                transformers_directory = "data/transformers",
                model = model,
                feature_names = feature_names, 
                output_folder = output_folder)


utils_general.create_map(pixel_size = 0.008333, 
                          x_data=x_data, 
                          y_data=y_data, 
                          z_data = z_data, 
                          map_path = 'data/training_results/MAP.tif')
