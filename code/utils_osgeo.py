import numpy as np
from osgeo import gdal
from scipy.io import savemat, loadmat
import os

def interpolate_raster_nodata(data, nodata_value, max_search_distance=100, smoothing_iterations=0):
    default_nodata_value = -9999
    driver = gdal.GetDriverByName('MEM')
    ds_temp = driver.Create('', data.shape[1], data.shape[0], 1, gdal.GDT_Float32)
    band_temp = ds_temp.GetRasterBand(1)

    band_temp.WriteArray(data)
    if nodata_value is not None and isinstance(nodata_value, (int, float)):
        band_temp.SetNoDataValue(nodata_value)
    else:
        print(f"Invalid no-data value: {nodata_value}. Using default: {default_nodata_value}")
        band_temp.SetNoDataValue(default_nodata_value)

    gdal.FillNodata(targetBand=band_temp, maskBand=None, maxSearchDist=max_search_distance, smoothingIterations=smoothing_iterations)
    
    interpolated_data = band_temp.ReadAsArray()
    return interpolated_data


def extract_data_from_raster(data, gt, mask_data, nodata_value, base_name, directory_outputs):
    valid_mask = mask_data == 1
    y_indices, x_indices = np.where(valid_mask)

    # Adjust x, y coordinates to point to the center of the pixel
    x_coords = gt[0] + (x_indices + 0.5) * gt[1] + (y_indices + 0.5) * gt[2]
    y_coords = gt[3] + (x_indices + 0.5) * gt[4] + (y_indices + 0.5) * gt[5]

    x_file_path = os.path.join(directory_outputs, "X.mat")
    y_file_path = os.path.join(directory_outputs, "Y.mat")

    if not os.path.exists(x_file_path):
        savemat(x_file_path, {'X': np.transpose(x_coords)}, do_compression=True)
    if not os.path.exists(y_file_path):
        savemat(y_file_path, {'Y': np.transpose(y_coords)}, do_compression=True)

    z_values = data[y_indices, x_indices].astype(np.float32)  # Convert to float before assigning np.nan
    z_values[z_values == nodata_value] = np.nan

    mat_file_path = os.path.join(directory_outputs, f"{base_name}.mat")
    savemat(mat_file_path, {base_name: z_values}, do_compression=True)
    
    mat_data = loadmat(mat_file_path)[base_name]
    mat_no_data_count = np.isnan(mat_data).sum()
    mat_array_size = mat_data.size
    
    return mat_no_data_count, mat_array_size
