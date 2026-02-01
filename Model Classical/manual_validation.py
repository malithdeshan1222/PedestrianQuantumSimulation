import numpy as np
import rasterio
from scipy import stats
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

def validate_results(predicted_file, observed_file, output_dir="./validation"):
    """
    Validate results by comparing predicted probability distribution with ground truth.
    
    Parameters:
    -----------
    predicted_file : str
        Path to the predicted probability distribution GeoTIFF
    observed_file : str
        Path to the observed ground truth GeoTIFF
    output_dir : str
        Directory to save validation outputs
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load predicted and observed data
    with rasterio.open(predicted_file) as src:
        predicted = src.read(1)
        meta = src.meta.copy()
    
    with rasterio.open(observed_file) as src:
        observed = src.read(1)
    
    # Ensure data are valid (handle nodata and masked values)
    if isinstance(predicted, np.ma.MaskedArray):
        predicted = predicted.filled(0)
    
    if isinstance(observed, np.ma.MaskedArray):
        observed = observed.filled(0)
    
    # Normalize if necessary
    predicted_sum = np.sum(predicted)
    observed_sum = np.sum(observed)
    
    if predicted_sum > 0:
        predicted = predicted / predicted_sum
        
    if observed_sum > 0:
        observed = observed / observed_sum
    
    # Calculate metrics
    # RMSE
    rmse = np.sqrt(mean_squared_error(observed.flatten(), predicted.flatten()))
    
    # Pearson's r
    pearson_r, p_value = stats.pearsonr(observed.flatten(), predicted.flatten())
    
    # Print metrics
    print(f"Validation Results:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  Pearson's r: {pearson_r:.6f}")
    print(f"  p-value: {p_value:.6e}")
    
    # Save metrics to file
    metrics_file = os.path.join(output_dir, "validation_metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write("Validation Metrics\n")
        f.write("=================\n\n")
        f.write(f"RMSE: {rmse:.6f}\n")
        f.write(f"Pearson's r: {pearson_r:.6f}\n")
        f.write(f"p-value: {p_value:.6e}\n")
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(observed.flatten(), predicted.flatten(), alpha=0.5)
    plt.title(f"Predicted vs Observed\nRMSE: {rmse:.4f}, Pearson's r: {pearson_r:.4f}")
    plt.xlabel("Observed Values")
    plt.ylabel("Predicted Values")
    
    # Add 1:1 line
    min_val = min(np.min(predicted), np.min(observed))
    max_val = max(np.max(predicted), np.max(observed))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Save plot
    scatter_file = os.path.join(output_dir, "validation_scatter.png")
    plt.savefig(scatter_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save difference map
    diff = np.abs(predicted - observed)
    
    diff_file = os.path.join(output_dir, "difference_map.tif")
    with rasterio.open(diff_file, 'w', **meta) as dst:
        dst.write(diff.astype(rasterio.float32), 1)
    
    # Also save as PNG for visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(diff, cmap='hot')
    plt.colorbar(label='Absolute Difference')
    plt.title('Difference Between Predicted and Observed')
    plt.axis('off')
    
    diff_png = os.path.join(output_dir, "difference_map.png")
    plt.savefig(diff_png, bbox_inches='tight', dpi=300)
    plt.close()
    
    return {
        "rmse": rmse,
        "pearson_r": pearson_r,
        "p_value": p_value
    }

if __name__ == "__main__":
    # Example usage
    predicted_file = "results/probability_distribution.tif"  # Update with your path
    observed_file = "D:/DATA/MALITH/Uni/Semester 08/ISRP/Model GIS/Model_VS_Qiskit/Model Classical/data/ground_truth.tif"  # Update with your path
    
    validate_results(predicted_file, observed_file, output_dir="./validation_results")