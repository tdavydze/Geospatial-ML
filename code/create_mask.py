import yaml
import numpy as np
import rasterio
import os
import logging

logging.basicConfig(level=logging.INFO)

def evaluate_condition(array, condition):
    return eval(f"array {condition}")


def create_mask(config):
    tif_folder = config['tif_folder']
    output_path = os.path.join(tif_folder, "Mask.tif")
    conditions = config.get('masking_conditions', [])
    regions = config.get('masking_regions', [])

    datasets = {}
    try:
        for condition in conditions:
            file_path = os.path.join(tif_folder, condition['file'])
            datasets[condition['file']] = rasterio.open(file_path)

        first_ds = next(iter(datasets.values()))
        mask_meta = first_ds.meta.copy()

        no_data_value = first_ds.nodatavals[0] or np.nan

        mask = np.ones_like(first_ds.read(1), dtype=np.float32)

        for condition in conditions:
            band = datasets[condition['file']].read(1)
            invalid_mask = evaluate_condition(band, condition['condition'])
            mask[invalid_mask] = no_data_value

        if regions:   #Handle regions specified by X and Y boundaries
            transform = first_ds.transform  # The affine transform object of the first dataset
            for region in regions:
                x_min, x_max = region['x_min'], region['x_max']
                y_min, y_max = region['y_min'], region['y_max']
    
                logging.info(f"Processing region: x: {x_min}-{x_max}, y: {y_min}-{y_max}")
                
                col_min, row_min = ~transform * (x_min, y_max)  #Note the switch of y_max and y_min due to the axis direction
                col_max, row_max = ~transform * (x_max, y_min) #Convert geographical coordinates to pixel coordinates
                
                row_min, row_max = min(row_min, row_max), max(row_min, row_max)                 #Correct the order if necessary
                col_min, col_max = min(col_min, col_max), max(col_min, col_max)                 #Correct the order if necessary
    
                row_min, row_max, col_min, col_max = map(int, [row_min, row_max, col_min, col_max])   #Convert to integers
    
                logging.info(f"Applying mask for pixel coordinates: row: {row_min}-{row_max}, col: {col_min}-{col_max}")
    
                mask[row_min:row_max, col_min:col_max] = no_data_value          #Update the mask

        mask_meta.update(dtype=rasterio.float32, nodata=no_data_value)
        with rasterio.open(output_path, 'w', **mask_meta) as dst:
            dst.write(mask, 1)

    finally:
        for ds in datasets.values():
            ds.close()
            
            
if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    create_mask(config)
