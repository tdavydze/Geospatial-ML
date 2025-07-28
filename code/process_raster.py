import os
import time
import yaml
from osgeo import gdal
from utils_osgeo import interpolate_raster_nodata, extract_data_from_raster  # Import the modified function

def process_tif_file(input_path, base_name, directory_outputs, mask_file_path, should_interpolate):
    mask_ds = gdal.Open(mask_file_path)
    mask_data = mask_ds.GetRasterBand(1).ReadAsArray()

    ds = gdal.Open(input_path)
    data = ds.GetRasterBand(1).ReadAsArray()
    nodata_value = ds.GetRasterBand(1).GetNoDataValue()

    original_no_data_count = (data == nodata_value).sum()

    if should_interpolate:
        data = interpolate_raster_nodata(data, nodata_value)

    gt = ds.GetGeoTransform()
    
    # Here, assume that the modified extract_data_from_raster function will return the required counts.
    mat_no_data_count, mat_array_size = extract_data_from_raster(data, gt, mask_data, nodata_value, base_name, directory_outputs)

    ds = None

    return original_no_data_count, data.size, mat_no_data_count, mat_array_size


def seconds_to_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return int(h), int(m), s


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    mask_file_path = os.path.join(config['tif_folder'], "Mask.tif")

    # Check for X.mat and Y.mat
    if os.path.exists(os.path.join(config['output_folder'], 'X.mat')) and os.path.exists(os.path.join(config['output_folder'], 'Y.mat')):
        print(f"X.mat and Y.mat have been created.")
    else:
        print(f"X.mat and Y.mat have not been created.")
    
    # Existing process for features.
    feature_configs = config.get('features', {})
    process_maps(feature_configs, config, mask_file_path, 'feature')
    
    # New process for other_maps.
    other_maps_configs = config.get('other_maps', {})
    process_maps(other_maps_configs, config, mask_file_path, 'other_map')


def process_maps(map_configs, config, mask_file_path, map_type):
    processed_tif_files = [
        f for f in map_configs if os.path.isfile(os.path.join(config['output_folder'], f"{f}.mat"))
    ]

    to_process_files = set(map_configs.keys()) - set(processed_tif_files)
    print(f"{len(processed_tif_files)} {map_type} maps have already been processed.")
    print(f"Processing {len(to_process_files)} remaining {map_type} maps...")

    start_time = time.time()

    for index, base_name in enumerate(to_process_files, 1):
        if base_name in ['X', 'Y']:
            continue

        print(f"\nProcessing {map_type} map {base_name}.tif ({index}/{len(to_process_files)})")
        input_path = os.path.join(config['tif_folder'], f"{base_name}.tif")
        should_interpolate = map_configs.get(base_name, {}).get('Interpolate', False)

        file_start_time = time.time()
        original_no_data_count, array_size, mat_no_data_count, mat_array_size = process_tif_file(
            input_path, base_name, config['output_folder'], mask_file_path, should_interpolate)
        file_elapsed_time = time.time() - file_start_time

        print(f"{map_type} map {base_name}.tif processed in {file_elapsed_time:.2f} seconds. "
              f"Original no-data values in .tif: {original_no_data_count}/{array_size}. "
              f"Remaining no-data values in .mat: {mat_no_data_count}/{mat_array_size}.")

        elapsed_time = time.time() - start_time
        avg_time_per_file = elapsed_time / index
        estimated_time_remaining = avg_time_per_file * (len(to_process_files) - index)

        h, m, s = seconds_to_hms(estimated_time_remaining)
        print(f"Total elapsed time: {elapsed_time:.2f} seconds. "
              f"Estimated time remaining: {h:d} hours {m:d} minutes {s:.2f} seconds.")
    
    print(f"{map_type.capitalize()} maps processing completed!")


if __name__ == "__main__":
    main()


