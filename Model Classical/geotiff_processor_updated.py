import numpy as np
import os
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
from urban_potential_field import UrbanPotentialField
import logging
from validation_metrics import calculate_validation_metrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeotiffProcessor:
    def __init__(self, output_dir="./output"):
        """
        Initialize GeotiffProcessor to handle GeoTIFF inputs for urban potential field analysis.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save outputs
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Dictionary to store raster data
        self.rasters = {}
        self.common_meta = None
        self.grid_shape = None
        self.ground_truth_data = None
    
    def load_geotiff(self, factor_name, filepath):
        """
        Load a GeoTIFF file and store it in the rasters dictionary.
        
        Parameters:
        -----------
        factor_name : str
            Name of the factor (e.g., 'building_density')
        filepath : str
            Path to the GeoTIFF file
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            with rasterio.open(filepath) as src:
                logger.info(f"Loading {factor_name} from {filepath}")
                # Read the first band (index 1)
                data = src.read(1, masked=True)
                
                # Store raster data and metadata
                self.rasters[factor_name] = {
                    'data': data,
                    'meta': src.meta.copy(),
                    'filepath': filepath
                }
                
                # If it's ground truth, store it separately for validation
                if factor_name == 'ground_truth':
                    self.ground_truth_data = data
                    logger.info("Ground truth data loaded successfully")
                
                # If first raster loaded, use its metadata as base
                if self.common_meta is None:
                    self.common_meta = src.meta.copy()
                    self.grid_shape = data.shape
                
                return True
        except RasterioIOError as e:
            logger.error(f"Error opening {filepath}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error processing {filepath}: {str(e)}")
            return False
    
    # ... [other methods from the original class] ...

    def process_potential_field(self, alpha=0.5, beta=0.5, temperature=0.1):
        """
        Process all rasters to generate the urban potential field and probability map.
        
        Parameters:
        -----------
        alpha : float
            Weight for V urbs
        beta : float
            Weight for V civitas
        temperature : float
            Temperature parameter in the Boltzmann distribution
        
        Returns:
        --------
        tuple
            (UrbanPotentialField object, success flag)
        """
        # ... [existing implementation] ...
        
        # Generate probability distribution
        probability_dist = upf.generate_probability_distribution(temperature=temperature)
        
        # Explicitly calculate and save validation metrics if ground truth is available
        if self.ground_truth_data is not None:
            logger.info("Ground truth data found - calculating validation metrics")
            ground_truth = self.ground_truth_data
            
            # Fill masked values (if any) with zeros
            if isinstance(ground_truth, np.ma.MaskedArray):
                ground_truth = ground_truth.filled(0)
            
            # Normalize ground truth (should sum to 1 like probability distribution)
            ground_truth = ground_truth / np.sum(ground_truth)
            
            # Calculate metrics
            metrics = calculate_validation_metrics(probability_dist, ground_truth)
            
            logger.info("Validation metrics:")
            logger.info(f"  RMSE: {metrics['rmse']:.6f}")
            logger.info(f"  Pearson's r: {metrics['pearson_r']:.6f}")
            logger.info(f"  p-value: {metrics['p_value']:.6e}")
            
            # Save metrics to file
            metrics_path = os.path.join(self.output_dir, "validation_metrics.txt")
            with open(metrics_path, 'w') as f:
                f.write("Validation Metrics\n")
                f.write("==================\n\n")
                f.write(f"RMSE: {metrics['rmse']:.6f}\n")
                f.write(f"Pearson's r: {metrics['pearson_r']:.6f}\n")
                f.write(f"p-value: {metrics['p_value']:.6e}\n")
            
            # Also generate a scatter plot of predicted vs observed
            self.generate_validation_plot(probability_dist, ground_truth)
        else:
            logger.warning("No ground truth data found - skipping validation metrics calculation")
        
        # ... [rest of the method] ...
    
    def generate_validation_plot(self, predicted, observed):
        """
        Generate a scatter plot comparing predicted and observed values.
        
        Parameters:
        -----------
        predicted : numpy.ndarray
            Predicted values (probability distribution)
        observed : numpy.ndarray
            Observed values (ground truth)
        """
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        
        # Flatten arrays for plotting
        pred_flat = predicted.flatten()
        obs_flat = observed.flatten()
        
        # Calculate metrics for plot title
        metrics = calculate_validation_metrics(predicted, observed)
        
        # Create scatter plot
        plt.scatter(obs_flat, pred_flat, alpha=0.5)
        plt.title(f"Predicted vs Observed\nRMSE: {metrics['rmse']:.4f}, Pearson's r: {metrics['pearson_r']:.4f}")
        plt.xlabel("Observed Values (Ground Truth)")
        plt.ylabel("Predicted Values (Probability Distribution)")
        
        # Add 1:1 line
        min_val = min(np.min(pred_flat), np.min(obs_flat))
        max_val = max(np.max(pred_flat), np.max(obs_flat))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "validation_scatter_plot.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"Validation scatter plot saved to {plot_path}")