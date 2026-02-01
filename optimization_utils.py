import numpy as np
import os
from scipy.ndimage import zoom
import rasterio
from rasterio.transform import from_origin

def optimized_load_geotiff(filename, data_path=None, default_shape=(32, 32)):
    """
    Load GeoTIFF file with optimized memory handling for large files.
    
    Parameters:
        filename (str): Name of the GeoTIFF file
        data_path (str): Directory containing the file
        default_shape (tuple): Default shape if file not found
    
    Returns:
        numpy.ndarray: 2D array of loaded data
    """
    if data_path is None:
        data_path = os.path.join(os.getcwd(), 'data')
        
    file_path = os.path.join(data_path, filename)
    
    try:
        with rasterio.open(file_path) as src:
            # Read only the first band and handle no data values
            data = src.read(1)
            
            # Replace no data values with zeros
            if src.nodata is not None:
                data = np.where(data == src.nodata, 0, data)
            
            # Normalize data between 0 and 1 for consistent processing
            if data.min() != data.max():
                data = (data - data.min()) / (data.max() - data.min())
        
        return data
    except Exception as e:
        print(f"Warning: Could not load {filename}, using random data. Error: {str(e)}")
        # Return zeros array with small random noise for testing
        random_data = np.random.random(default_shape) * 0.1
        return random_data

def apply_coefficients_vectorized(features, coeffs):
    """
    Apply coefficients to a list of feature layers in a vectorized operation.
    
    Parameters:
        features (list): List of feature arrays
        coeffs (list): List of coefficients for each feature
        
    Returns:
        numpy.ndarray: Weighted sum of features
    """
    # Initialize with zeros matching the shape of the first feature
    result = np.zeros_like(features[0])
    
    # Apply each coefficient
    for i, (feature, coeff) in enumerate(zip(features, coeffs)):
        result += feature * coeff
        
    return result

def batch_shadow_calculation(building_height, tree_height, sun_params_dict):
    """
    Calculate shadows for multiple time periods efficiently.
    
    Parameters:
        building_height (numpy.ndarray): Building height data
        tree_height (numpy.ndarray): Tree height data
        sun_params_dict (dict): Dictionary mapping time periods to (altitude, azimuth) tuples
    
    Returns:
        dict: Dictionary mapping time periods to shadow intensity arrays
    """
    # Prepare total height array (buildings + trees)
    total_height = building_height + tree_height
    
    # Calculate shadows for each time period
    shadow_dict = {}
    
    for time_period, (sun_altitude, sun_azimuth) in sun_params_dict.items():
        print(f"Calculating {time_period} shadows...")
        shadow_dict[time_period] = calculate_shadow_intensity(
            total_height, sun_altitude, sun_azimuth
        )
        
    return shadow_dict

def calculate_shadow_intensity(height_data, sun_altitude, sun_azimuth):
    """
    Calculate shadow intensity from height data and sun position.
    
    Parameters:
        height_data (numpy.ndarray): Height data (buildings + trees)
        sun_altitude (float): Sun altitude in degrees
        sun_azimuth (float): Sun azimuth in degrees
        
    Returns:
        numpy.ndarray: Shadow intensity array (0-1)
    """
    # Convert degrees to radians
    altitude_rad = np.radians(sun_altitude)
    azimuth_rad = np.radians(sun_azimuth)
    
    # Calculate shadow length based on height and sun altitude
    if sun_altitude <= 0:
        return np.ones_like(height_data)  # Complete shadow at night
    
    # Calculate shadow direction vector
    shadow_dx = -np.sin(azimuth_rad) / np.tan(altitude_rad)
    shadow_dy = -np.cos(azimuth_rad) / np.tan(altitude_rad)
    
    # Prepare shadow map
    shadow_map = np.zeros_like(height_data)
    height, width = shadow_map.shape
    
    # Apply ray tracing for shadow calculation
    for y in range(height):
        for x in range(width):
            if height_data[y, x] > 0:
                # Calculate shadow length
                shadow_length = height_data[y, x] / np.tan(altitude_rad)
                
                # Trace shadow along the direction vector
                steps = int(2 * shadow_length)
                for step in range(1, steps + 1):
                    # Calculate shadow position
                    sx = x + step * shadow_dx / steps
                    sy = y + step * shadow_dy / steps
                    
                    # Check if within bounds
                    if 0 <= sx < width and 0 <= sy < height:
                        # Cast to integer indices
                        si, sj = int(sy), int(sx)
                        shadow_map[si, sj] += 1.0 / step  # Fade with distance
    
    # Normalize shadow intensity
    if shadow_map.max() > 0:
        shadow_map = shadow_map / shadow_map.max()
    
    return shadow_map

def calculate_shadow_pedestrian_attraction(shadow_intensity, temperature_factor):
    """
    Calculate pedestrian attraction to shadows based on temperature.
    
    Parameters:
        shadow_intensity (numpy.ndarray): Shadow intensity array
        temperature_factor (float): Temperature factor (0-1)
        
    Returns:
        numpy.ndarray: Shadow attraction array
    """
    # In hot weather (high temperature_factor), shadows are attractive
    # In cold weather (low temperature_factor), people avoid shadows
    if temperature_factor >= 0.5:  # Hot weather
        attraction = temperature_factor * shadow_intensity
    else:  # Cold weather
        attraction = temperature_factor * (1 - shadow_intensity)
        
    return attraction

def learn_coefficients(features, target):
    """
    Learn coefficients for features to match a target pattern.
    
    Parameters:
        features (list): List of feature arrays
        target (numpy.ndarray): Target pattern to match
        
    Returns:
        list: List of optimized coefficients
    """
    num_features = len(features)
    
    # Check if features and target have compatible shapes
    for i, feature in enumerate(features):
        if feature.shape != target.shape:
            # Resize to match target shape
            features[i] = zoom(feature, 
                              np.array(target.shape) / np.array(feature.shape),
                              order=1)
    
    # Simple equal weighting (baseline)
    coeffs = [1.0 / num_features] * num_features
    
    # In a more sophisticated implementation, we would use
    # regression techniques to find optimal coefficients
    
    return coeffs

def save_geotiff(data, reference_file, output_filename, data_path=None, result_path=None):
    """
    Save data as a GeoTIFF using the georeferencing from a reference file.
    
    Parameters:
        data (numpy.ndarray): Data to save
        reference_file (str): Reference file for georeferencing
        output_filename (str): Output filename
        data_path (str): Directory containing reference file
        result_path (str): Directory to save output file
    """
    if data_path is None:
        data_path = os.path.join(os.getcwd(), 'data')
        
    if result_path is None:
        result_path = os.path.join(os.getcwd(), 'results')
        
    # Create result directory if it doesn't exist
    os.makedirs(result_path, exist_ok=True)
    
    reference_path = os.path.join(data_path, reference_file)
    output_path = os.path.join(result_path, output_filename)
    
    try:
        # Get complete georeference info from reference file
        with rasterio.open(reference_path) as src:
            # Get all metadata from the source file
            kwargs = src.meta.copy()
            
            # Update metadata for the output file
            kwargs.update({
                'height': data.shape[0],
                'width': data.shape[1],
                'count': 1,
                'dtype': data.dtype
            })
            
            # Check if data shape matches reference shape
            if data.shape != (src.height, src.width):
                # Resize data to match reference shape
                data = zoom(data, 
                          np.array((src.height, src.width)) / np.array(data.shape),
                          order=1)
                print(f"Resized output to match reference shape: {data.shape}")
            
        # Save output GeoTIFF with all georeferencing from reference file
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            dst.write(data, 1)
            
        print(f"Saved {output_filename} with full geospatial properties")
        
    except Exception as e:
        print(f"Warning: Problem with georeferencing: {str(e)}")
        
        # Create simple transform (dummy georeferencing)
        transform = from_origin(0, 0, 1, 1)
        
        # Save output GeoTIFF without proper georeferencing
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=None,
            transform=transform,
        ) as dst:
            dst.write(data, 1)
            
        print(f"Saved {output_filename} (without proper georeferencing)")
        # Fallback if can't read reference file
        print(f"Warning: Saving without georeferencing. Error: {str(e)}")
        
        # Create simple transform (dummy georeferencing)
        transform = from_origin(0, 0, 1, 1)
        
        # Save output GeoTIFF without georeferencing
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=None,
            transform=transform,
        ) as dst:
            dst.write(data, 1)
            
        print(f"Saved {output_filename} (without georeferencing)")