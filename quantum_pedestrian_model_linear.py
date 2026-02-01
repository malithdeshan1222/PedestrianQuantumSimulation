import os
import numpy as np
import rasterio

# =============================================================================
# Utility Functions
# =============================================================================

def load_geotiff(filename):
    """Load GeoTIFF with proper handling of spatial reference."""
    try:
        with rasterio.open(filename) as src:
            data = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs
            return data, transform, crs
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None, None, None

def save_geotiff(data, output_filename, transform, crs):
    """Save data as GeoTIFF with spatial reference."""
    try:
        with rasterio.open(
            output_filename,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(data, 1)
            print(f"Saved {output_filename}")
    except Exception as e:
        print(f"Error saving {output_filename}: {e}")

def normalize(data):
    """Normalize data to the range [0, 1]."""
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def rescale(data, original_min, original_max):
    """Rescale data back to the original range."""
    return data * (original_max - original_min) + original_min

# =============================================================================
# Matrix Algebra Model
# =============================================================================

def train_matrix_model(potential_fields, probability_fields):
    """
    Train a matrix algebra model to map potential fields to probability fields.
    Args:
        potential_fields: List of 2D numpy arrays representing potential fields.
        probability_fields: List of 2D numpy arrays representing probability fields.
    Returns:
        W: Weight matrix that maps potential fields to probability fields.
        normalization_params: Dictionary containing normalization parameters for inputs and outputs.
    """
    # Normalize inputs and outputs
    X_normalized = [normalize(field).flatten() for field in potential_fields]
    Y_normalized = [normalize(field).flatten() for field in probability_fields]
    X = np.array(X_normalized).T  # Transpose to shape (num_pixels, num_features)
    Y = np.array(Y_normalized).T  # Transpose to shape (num_pixels, num_outputs)

    # Compute the weight matrix W using the Moore-Penrose pseudoinverse
    W = np.linalg.pinv(X) @ Y
    print("Weight matrix W computed using matrix algebra.")

    # Store normalization parameters
    normalization_params = {
        "X_min": [np.min(field) for field in potential_fields],
        "X_max": [np.max(field) for field in potential_fields],
        "Y_min": [np.min(field) for field in probability_fields],
        "Y_max": [np.max(field) for field in probability_fields],
    }

    return W, normalization_params

def simulate_with_matrix_model(W, new_potential_field, shape, normalization_params, reference_fields=None):
    """
    Simulate the probability raster for a new potential field using the trained weight matrix.
    Args:
        W: Weight matrix that maps potential fields to probability fields.
        new_potential_field: 2D numpy array representing the new potential field.
        shape: Tuple representing the original shape of the raster.
        normalization_params: Dictionary containing normalization parameters for inputs and outputs.
        reference_fields: Optional list of 2D numpy arrays representing additional potential fields
                          to match the model's input features.
                          If None, the new potential field will be repeated to create 3 features.
    Returns:
        simulated_probabilities: List of 2D numpy arrays representing multiple output features.
    """
    # Normalize the new potential field
    flat_new_field = normalize(new_potential_field).flatten()

    if reference_fields:
        # Normalize and flatten the reference fields
        flat_references = [normalize(field).flatten() for field in reference_fields]
        # Combine the new field with the reference fields
        stacked_field = np.stack([flat_new_field] + flat_references, axis=1)
    else:
        # Repeat the new potential field to create the required number of features
        stacked_field = np.stack([flat_new_field] * W.shape[0], axis=1)

    # Predict the probability field
    simulated_flat = stacked_field @ W

    # Rescale simulated outputs to the original range
    simulated_rescaled = [
        rescale(simulated_flat[:, i], normalization_params["Y_min"][i], normalization_params["Y_max"][i]).reshape(shape)
        for i in range(simulated_flat.shape[1])
    ]

    return simulated_rescaled

# =============================================================================
# Main Application
# =============================================================================

def main():
    # Define folder paths and file paths
    folder = "Scenario_Inputs"
    new_potential_field_path = r"D:\DATA\MALITH\Uni\Semester 08\ISRP\Model GIS\Model_VS_Qiskit\Scenario_Inputs\V_T_Edited_S1.tif"
    output_path = r"D:\DATA\MALITH\Uni\Semester 08\ISRP\Model GIS\Model_VS_Qiskit\Scenario_Inputs\S1_output.tif"

    # Get all files in the folder
    files = os.listdir(folder)
    
    # Separate potential fields and probability rasters based on naming convention
    potential_files = sorted([f for f in files if f.startswith("V_total_") and f.endswith(".tif")])
    probability_files = sorted([f for f in files if f.startswith("P_P_") and f.endswith(".tif")])
    
    # Ensure that the number of potential fields matches the number of probability rasters
    if len(potential_files) != len(probability_files):
        print("Error: Mismatch in the number of potential fields and probability rasters.")
        return

    # Load potential and probability fields
    potential_fields, probability_fields = [], []
    transform, crs = None, None
    for p_file, prob_file in zip(potential_files, probability_files):
        # Load potential field
        potential_field, transform, crs = load_geotiff(os.path.join(folder, p_file))
        # Load probability field
        probability_field, _, _ = load_geotiff(os.path.join(folder, prob_file))
        
        if potential_field is not None and probability_field is not None:
            potential_fields.append(potential_field)
            probability_fields.append(probability_field)

    # Train the model
    W, normalization_params = train_matrix_model(potential_fields, probability_fields)

    # Load the new potential field
    new_potential_field, new_transform, new_crs = load_geotiff(new_potential_field_path)
    if new_potential_field is None:
        print("Failed to load the new potential field. Exiting.")
        return

    # Use reference fields if available
    reference_fields = potential_fields[1:]  # Example: use evening and afternoon fields
    simulated_probabilities = simulate_with_matrix_model(
        W, new_potential_field, new_potential_field.shape, normalization_params, reference_fields
    )

    # Combine outputs by averaging
    combined_probability = np.mean(np.array(simulated_probabilities), axis=0)

    # Save the combined probability raster
    save_geotiff(combined_probability, output_path, new_transform or transform, new_crs or crs)

    # Optionally save all outputs separately
    for i, prob in enumerate(simulated_probabilities):
        save_geotiff(
            prob,
            output_path.replace(".tif", f"_output_{i+1}.tif"),
            new_transform or transform,
            new_crs or crs,
        )

if __name__ == "__main__":
    main()