import os
import numpy as np
from optimization_utils import optimized_load_geotiff

data_path = "data"
files = ["building_density.tif", "street_centrality.tif", "pedestrian_density.tif"]

for f in files:
    try:
        data = optimized_load_geotiff(f, data_path=data_path)
        print(f"{f}: {data.shape}, min={data.min():.2f}, max={data.max():.2f}")
    except Exception as e:
        print(f"Error loading {f}: {e}")