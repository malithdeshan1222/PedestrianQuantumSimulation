#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classical implementation of pedestrian probability model using multiple urban factors.
This module processes multiple urban factors to generate pedestrian probability maps
and validation metrics.
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import os
from scipy import optimize
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("classical_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ClassicalModel")


def optimized_load_geotiff(filename, data_path=None):
    """
    Load a GeoTIFF file from the specified path.
    
    Parameters:
    -----------
    filename : str
        Name of the GeoTIFF file
    data_path : str or Path, optional
        Path to the directory containing the data
    
    Returns:
    --------
    numpy.ndarray
        The raster data
    """
    if data_path is None:
        data_path = Path("data")
    else:
        data_path = Path(data_path)
    
    file_path = data_path / filename
    
    try:
        with rasterio.open(file_path) as src:
            return src.read(1)
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise


class ClassicalPedestrianModel:
    """
    Classical model for calculating pedestrian probability maps based on multiple urban factors.
    """
    
    def __init__(self, output_dir='./outputs', config=None):
        """
        Initialize the model.
        
        Parameters:
        -----------
        output_dir : str
            Directory where output files will be saved
        config : object, optional
            Configuration object with data paths and other settings
        """
        self.factors = {}
        self.coefficients = None
        self.potential_field = None
        self.energy = None
        self.probability_map = None
        self.enhanced_probability = None
        self.output_dir = output_dir
        self.config = config
        self.metadata = None
        self.factor_names = []
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def load_factors(self, data_path=None):
        """
        Load all urban factor raster data.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to directory containing the raster files
        """
        factor_files = [
            "building_density.tif",
            "street_centrality.tif",
            "building_height.tif",
            "tree_height.tif",
            "poi_density.tif",
            "wall_constraint.tif",
            "mean_depth.tif",
            "isovist.tif",
            "vehicle_density.tif",
            "pedestrian_accessibility.tif"
        ]
        
        self.pedestrian_density = None  # For validation
        
        logger.info(f"Loading {len(factor_files)} urban factors...")
        start_time = time.time()
        
        # Get data path from config if available
        if data_path is None and self.config is not None:
            data_path = self.config.data.data_path
            
        # Load each factor
        first_file = True
        for filename in factor_files:
            factor_name = filename.split('.')[0]
            self.factor_names.append(factor_name)
            
            try:
                raster_data = optimized_load_geotiff(filename, data_path)
                
                # Store metadata from the first raster
                if first_file:
                    with rasterio.open(os.path.join(data_path, filename)) as src:
                        self.metadata = src.meta
                    first_file = False
                
                # Store the factor data
                self.factors[factor_name] = raster_data
                logger.info(f"Loaded {factor_name} with shape {raster_data.shape}")
                
            except Exception as e:
                logger.error(f"Failed to load {factor_name}: {e}")
                raise
        
        # Load pedestrian density for validation
        try:
            self.pedestrian_density = optimized_load_geotiff("pedestrian_density.tif", data_path)
            logger.info(f"Loaded pedestrian_density with shape {self.pedestrian_density.shape}")
        except Exception as e:
            logger.warning(f"Could not load pedestrian_density.tif: {e}. Validation will be skipped.")
        
        # Ensure all rasters have the same shape
        shape = None
        for name, factor in self.factors.items():
            if shape is None:
                shape = factor.shape
            elif shape != factor.shape:
                raise ValueError(f"Factor {name} has shape {factor.shape}, expected {shape}")
        
        logger.info(f"Loaded all factors in {time.time() - start_time:.2f} seconds")
    
    def _normalize_factor(self, factor_array):
        """
        Normalize factor values to range [0, 1]
        
        Parameters:
        -----------
        factor_array : numpy.ndarray
            The factor data to normalize
            
        Returns:
        --------
        numpy.ndarray
            Normalized factor data
        """
        min_val = np.nanmin(factor_array)
        max_val = np.nanmax(factor_array)
        if max_val - min_val > 0:
            return (factor_array - min_val) / (max_val - min_val)
        else:
            # Handle case where all values are the same
            return np.zeros_like(factor_array)
    
    def normalize_factors(self):
        """
        Normalize all factors to have values in range [0, 1]
        """
        logger.info("Normalizing all factors...")
        for name in self.factors:
            # Skip factors that are already normalized or are binary masks
            if name == 'wall_constraint':
                continue
                
            self.factors[name] = self._normalize_factor(self.factors[name])
            logger.info(f"Normalized {name}")
    
    def calculate_potential_field(self, coefficients=None):
        """
        Calculate the potential field based on all factors with given coefficients.
        
        Parameters:
        -----------
        coefficients : array-like, optional
            Coefficients for each factor. If None, use equal weights.
        
        Returns:
        --------
        numpy.ndarray
            The calculated potential field
        """
        if not self.factors:
            raise ValueError("No factors loaded. Call load_factors() first.")
        
        # If no coefficients provided, use equal weights
        if coefficients is None:
            coefficients = np.ones(len(self.factors))
        
        # Check if we have correct number of coefficients
        if len(coefficients) != len(self.factors):
            raise ValueError(f"Expected {len(self.factors)} coefficients, got {len(coefficients)}")
        
        # Initialize potential field with zeros
        potential = np.zeros_like(list(self.factors.values())[0], dtype=float)
        
        # Calculate linear combination of factors
        for i, (name, factor) in enumerate(self.factors.items()):
            coef = coefficients[i]
            
            # Special handling for wall_constraint which is a mask
            if name == 'wall_constraint':
                # Apply wall constraint as a mask (0 where walls exist)
                potential += coef * (1 - factor)  # Invert the wall constraint
            else:
                potential += coef * factor
        
        return potential
    
    def objective_function(self, coefficients, observed_data=None):
        """
        Objective function for coefficient optimization.
        
        Parameters:
        -----------
        coefficients : array-like
            Coefficients for each factor
        observed_data : numpy.ndarray, optional
            Observed pedestrian density for validation
        
        Returns:
        --------
        float
            The error metric (RMSE) if observed_data is provided,
            otherwise a regularization term
        """
        potential = self.calculate_potential_field(coefficients)
        
        # Calculate energy
        energy = -potential
        
        # Apply Boltzmann distribution
        # Use mean of absolute energy as temperature parameter
        temperature = np.mean(np.abs(energy))
        boltzmann = np.exp(-energy / temperature)
        
        # Normalize to get probability distribution
        probability = boltzmann / np.nansum(boltzmann)
        
        if observed_data is not None:
            # Calculate RMSE between model prediction and observed data
            # Exclude NaN values
            mask = ~(np.isnan(observed_data) | np.isnan(probability))
            if not np.any(mask):
                return 1e10  # Return a large value if no valid data
                
            return np.sqrt(mean_squared_error(
                observed_data[mask], 
                probability[mask]
            ))
        else:
            # Apply regularization to prevent extreme coefficient values
            reg_term = 0.01 * np.sum(coefficients**2)
            
            # Add constraint to make coefficients sum to approximately 1
            sum_penalty = 0.1 * abs(np.sum(coefficients) - 1.0)
            
            return reg_term + sum_penalty
    
    def optimize_coefficients(self, observed_data=None, initial_guess=None, method='Nelder-Mead'):
        """
        Optimize the coefficients of all factors using classical optimization.
        
        Parameters:
        -----------
        observed_data : numpy.ndarray, optional
            Observed pedestrian density for validation-based optimization
        initial_guess : list, optional
            Initial guess for coefficients
        method : str
            Optimization method to use
        
        Returns:
        --------
        numpy.ndarray
            Optimized coefficients
        """
        logger.info("Optimizing coefficients...")
        start_time = time.time()
        
        n_factors = len(self.factors)
        
        # Set initial guess if not provided
        if initial_guess is None:
            initial_guess = np.ones(n_factors) / n_factors  # Equal weights
        
        # Bounds to ensure coefficients are non-negative
        bounds = [(0, None) for _ in range(n_factors)]
        
        # Define the objective function for optimization
        def obj_func(x):
            return self.objective_function(x, observed_data)
        
        # Optimize using chosen method
        result = optimize.minimize(
            obj_func,
            initial_guess,
            method=method,
            bounds=bounds,
            options={
                'maxiter': 1000, 
                'disp': True
            }
        )
        
        self.coefficients = result.x
        
        # Normalize coefficients so they sum to 1
        self.coefficients = self.coefficients / np.sum(self.coefficients)
        
        # Log the optimized coefficients
        coef_str = "\n".join([f"{name}: {coef:.6f}" 
                              for name, coef in zip(self.factors.keys(), self.coefficients)])
        logger.info(f"Optimized coefficients:\n{coef_str}")
        logger.info(f"Optimization completed in {time.time() - start_time:.2f} seconds")
        
        return self.coefficients
    
    def calculate_energy(self):
        """
        Calculate the energy based on the optimized potential field.
        
        Returns:
        --------
        numpy.ndarray
            The calculated energy field
        """
        logger.info("Calculating energy field...")
        
        if self.coefficients is None:
            logger.warning("Coefficients not optimized, using equal weights")
            n_factors = len(self.factors)
            self.coefficients = np.ones(n_factors) / n_factors
        
        # Calculate potential field with optimized coefficients
        self.potential_field = self.calculate_potential_field(self.coefficients)
        
        # Energy is negative of potential
        self.energy = -self.potential_field
        
        logger.info("Energy field calculated")
        return self.energy
    
    def apply_boltzmann_distribution(self):
        """
        Apply Boltzmann distribution to calculate probability.
        
        Returns:
        --------
        numpy.ndarray
            The probability distribution
        """
        logger.info("Applying Boltzmann distribution...")
        
        if self.energy is None:
            self.calculate_energy()
        
        # Use mean of absolute energy as temperature parameter
        temperature = np.mean(np.abs(self.energy))
        boltzmann = np.exp(-self.energy / temperature)
        
        # Normalize to get probability distribution
        self.probability_map = boltzmann / np.nansum(boltzmann)
        
        logger.info("Boltzmann distribution applied")
        return self.probability_map
    
    def enhance_distribution(self, enhancement_factor=1.5):
        """
        Enhance the probability distribution to highlight areas of high probability.
        
        Parameters:
        -----------
        enhancement_factor : float
            Factor to enhance contrast in probability map
        
        Returns:
        --------
        numpy.ndarray
            The enhanced probability distribution
        """
        logger.info(f"Enhancing distribution with factor {enhancement_factor}...")
        
        if self.probability_map is None:
            self.apply_boltzmann_distribution()
        
        # Apply power transformation to enhance contrast
        enhanced = np.power(self.probability_map, enhancement_factor)
        
        # Re-normalize
        self.enhanced_probability = enhanced / np.nansum(enhanced)
        
        logger.info("Distribution enhanced")
        return self.enhanced_probability
    
    def validate(self, observed_data=None):
        """
        Calculate validation metrics for the model.
        
        Parameters:
        -----------
        observed_data : numpy.ndarray, optional
            Observed pedestrian density for validation
        
        Returns:
        --------
        dict
            Dictionary containing validation metrics
        """
        logger.info("Calculating validation metrics...")
        
        if self.enhanced_probability is None:
            self.enhance_distribution()
        
        # Use stored pedestrian density if not provided
        if observed_data is None:
            if self.pedestrian_density is None:
                logger.warning("No observed data available for validation")
                return None
            observed_data = self.pedestrian_density
        
        # Ensure observed data has the same shape
        if observed_data.shape != self.enhanced_probability.shape:
            logger.error("Observed data shape mismatch")
            raise ValueError("Observed data must have the same shape as model outputs")
        
        # Create mask for valid data (non-NaN values in both arrays)
        mask = ~(np.isnan(observed_data) | np.isnan(self.enhanced_probability))
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(
            observed_data[mask], 
            self.enhanced_probability[mask]
        ))
        
        # Calculate Pearson correlation
        # Flatten arrays for correlation calculation
        flat_observed = observed_data[mask].flatten()
        flat_predicted = self.enhanced_probability[mask].flatten()
        
        correlation, p_value = pearsonr(flat_observed, flat_predicted)
        
        validation_results = {
            'RMSE': rmse,
            'Pearson_r': correlation,
            'p_value': p_value
        }
        
        logger.info(f"Validation results:")
        logger.info(f"RMSE: {rmse:.6f}")
        logger.info(f"Spatial correlation (Pearson r): {correlation:.6f} (p-value: {p_value:.6f})")
        
        return validation_results
    
    def save_outputs(self):
        """
        Save the model outputs to files.
        """
        logger.info("Saving model outputs...")
        
        if self.enhanced_probability is None:
            self.enhance_distribution()
        
        # Save probability map as raster
        output_path = os.path.join(self.output_dir, 'pedestrian_probability.tif')
        
        # Update metadata for output
        out_meta = self.metadata.copy() if self.metadata else None
        
        if out_meta:
            out_meta.update({
                "driver": "GTiff",
                "dtype": "float32",
                "compress": "lzw"
            })
            
            with rasterio.open(output_path, 'w', **out_meta) as dst:
                dst.write(self.enhanced_probability.astype(np.float32), 1)
            
            logger.info(f"Saved probability map to {output_path}")
        else:
            logger.warning("No metadata available, can't save probability map as GeoTIFF")
        
        # Save coefficient values
        if self.coefficients is not None:
            coef_path = os.path.join(self.output_dir, 'factor_coefficients.csv')
            with open(coef_path, 'w') as f:
                f.write("factor,coefficient\n")
                for name, coef in zip(self.factors.keys(), self.coefficients):
                    f.write(f"{name},{coef:.6f}\n")
            logger.info(f"Saved factor coefficients to {coef_path}")
        
        # Save visualization as PNG
        plt.figure(figsize=(12, 10))
        plt.imshow(self.enhanced_probability, cmap='hot')
        plt.colorbar(label='Pedestrian Probability')
        plt.title('Pedestrian Probability Map (Classical Model)')
        viz_path = os.path.join(self.output_dir, 'pedestrian_probability.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved visualization to {viz_path}")
        
        # Create comparison plot if observed data is available
        if self.pedestrian_density is not None:
            plt.figure(figsize=(18, 6))
            
            plt.subplot(131)
            plt.imshow(self.pedestrian_density, cmap='viridis')
            plt.colorbar(label='Density')
            plt.title('Observed Pedestrian Density')
            
            plt.subplot(132)
            plt.imshow(self.enhanced_probability, cmap='hot')
            plt.colorbar(label='Probability')
            plt.title('Predicted Pedestrian Probability')
            
            plt.subplot(133)
            error = self.enhanced_probability - self._normalize_factor(self.pedestrian_density)
            plt.imshow(error, cmap='coolwarm', vmin=-0.5, vmax=0.5)
            plt.colorbar(label='Difference')
            plt.title('Error (Predicted - Observed)')
            
            plt.tight_layout()
            comp_path = os.path.join(self.output_dir, 'comparison.png')
            plt.savefig(comp_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved comparison plot to {comp_path}")
        
        # Save validation metrics if available
        validation = self.validate()
        if validation:
            val_path = os.path.join(self.output_dir, 'validation_metrics.txt')
            with open(val_path, 'w') as f:
                for metric, value in validation.items():
                    f.write(f"{metric}: {value:.6f}\n")
            logger.info(f"Saved validation metrics to {val_path}")
        
        logger.info("All outputs saved")
    
    def run_model(self, data_path=None, observed_data=None,
                  enhancement_factor=1.5, save_results=True):
        """
        Run the complete classical pedestrian probability model.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to directory containing the raster files
        observed_data : numpy.ndarray, optional
            Observed pedestrian density for validation
        enhancement_factor : float
            Factor to enhance contrast in probability map
        save_results : bool
            Whether to save outputs to files
            
        Returns:
        --------
        dict
            Dictionary containing model outputs and validation metrics
        """
        start_time = time.time()
        logger.info("Starting classical pedestrian model...")
        
        # Load data
        self.load_factors(data_path)
        
        # Normalize factors
        self.normalize_factors()
        
        # Use provided observed_data or the loaded pedestrian_density
        if observed_data is None and self.pedestrian_density is not None:
            observed_data = self.pedestrian_density
        
        # Optimize coefficients
        self.optimize_coefficients(observed_data)
        
        # Calculate energy
        self.calculate_energy()
        
        # Apply Boltzmann distribution
        self.apply_boltzmann_distribution()
        
        # Enhance distribution
        self.enhance_distribution(enhancement_factor)
        
        # Save outputs
        if save_results:
            self.save_outputs()
        
        # Validate if observed data is provided
        validation_metrics = None
        if observed_data is not None:
            validation_metrics = self.validate(observed_data)
        
        total_time = time.time() - start_time
        logger.info(f"Model execution completed in {total_time:.2f} seconds")
        
        return {
            'coefficients': self.coefficients,
            'coefficient_dict': dict(zip(self.factor_names, self.coefficients)),
            'probability_map': self.enhanced_probability,
            'validation': validation_metrics
        }


def main():
    """
    Example usage of the ClassicalPedestrianModel.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Classical Pedestrian Model')
    parser.add_argument('--data_path', type=str, default='data',
                        help='Path to directory containing raster data')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs')
    parser.add_argument('--enhancement', type=float, default=1.5,
                        help='Enhancement factor for probability distribution')
    
    args = parser.parse_args()
    
    # Create model instance
    model = ClassicalPedestrianModel(output_dir=args.output_dir)
    
    # Run the model
    results = model.run_model(
        data_path=args.data_path,
        enhancement_factor=args.enhancement
    )
    
    # Print summary of results
    print("\nModel Results Summary:")
    print("-----------------------")
    print("Optimized Factor Coefficients:")
    for factor, coef in zip(model.factor_names, results['coefficients']):
        print(f"  {factor}: {coef:.6f}")
    
    if results['validation']:
        print("\nValidation Metrics:")
        print(f"  RMSE: {results['validation']['RMSE']:.6f}")
        print(f"  Spatial correlation (Pearson r): {results['validation']['Pearson_r']:.6f}")
    
    print(f"\nOutputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()