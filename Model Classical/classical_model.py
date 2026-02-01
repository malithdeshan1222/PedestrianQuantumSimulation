#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classical implementation of the Quantum-Model for pedestrian probability calculation.
This module processes V urbs and V civitas rasters to generate pedestrian probability maps
and validation metrics.
"""

import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy import optimize
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import os


class ClassicalPedestrianModel:
    """
    A classical model for calculating pedestrian probability maps based on
    urban (V urbs) and civic (V civitas) factors.
    """
    
    def __init__(self, output_dir='./outputs'):
        """
        Initialize the model.
        
        Parameters:
        -----------
        output_dir : str
            Directory where output files will be saved
        """
        self.v_urbs = None
        self.v_civitas = None
        self.coefficients = None
        self.potential_field = None
        self.energy = None
        self.probability_map = None
        self.enhanced_probability = None
        self.output_dir = output_dir
        self.metadata = None
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def load_rasters(self, v_urbs_path, v_civitas_path):
        """
        Load V urbs and V civitas raster data.
        
        Parameters:
        -----------
        v_urbs_path : str
            Path to V urbs raster file
        v_civitas_path : str
            Path to V civitas raster file
        """
        with rasterio.open(v_urbs_path) as src:
            self.v_urbs = src.read(1)
            self.metadata = src.meta
        
        with rasterio.open(v_civitas_path) as src:
            self.v_civitas = src.read(1)
            
        # Ensure both rasters have the same shape
        if self.v_urbs.shape != self.v_civitas.shape:
            raise ValueError("Input rasters must have the same dimensions")
        
        print(f"Loaded rasters with shape: {self.v_urbs.shape}")
    
    def calculate_potential_field(self, coef_urbs=1.0, coef_civitas=1.0):
        """
        Calculate the potential field based on V urbs and V civitas with given coefficients.
        
        Parameters:
        -----------
        coef_urbs : float
            Coefficient for V urbs
        coef_civitas : float
            Coefficient for V civitas
        
        Returns:
        --------
        numpy.ndarray
            The calculated potential field
        """
        if self.v_urbs is None or self.v_civitas is None:
            raise ValueError("Rasters must be loaded first")
        
        # Calculate linear combination of factors
        potential = coef_urbs * self.v_urbs + coef_civitas * self.v_civitas
        return potential
    
    def objective_function(self, coefficients, observed_data=None):
        """
        Objective function for coefficient optimization.
        
        Parameters:
        -----------
        coefficients : array-like
            Coefficients [coef_urbs, coef_civitas]
        observed_data : numpy.ndarray, optional
            Observed pedestrian density for validation
        
        Returns:
        --------
        float
            The error metric (RMSE) if observed_data is provided, 
            otherwise a regularization term
        """
        coef_urbs, coef_civitas = coefficients
        potential = self.calculate_potential_field(coef_urbs, coef_civitas)
        
        # Calculate energy
        energy = -potential
        
        # Apply Boltzmann distribution
        boltzmann = np.exp(-energy / np.mean(np.abs(energy)))
        probability = boltzmann / np.sum(boltzmann)
        
        if observed_data is not None:
            # Calculate RMSE between model prediction and observed data
            return np.sqrt(mean_squared_error(observed_data, probability))
        else:
            # Simple regularization to avoid extreme coefficient values
            return 0.01 * (coef_urbs**2 + coef_civitas**2)
    
    def optimize_coefficients(self, observed_data=None, initial_guess=None):
        """
        Optimize the coefficients of V urbs and V civitas using classical optimization.
        
        Parameters:
        -----------
        observed_data : numpy.ndarray, optional
            Observed pedestrian density for validation-based optimization
        initial_guess : list, optional
            Initial guess for [coef_urbs, coef_civitas]
        
        Returns:
        --------
        tuple
            Optimized coefficients (coef_urbs, coef_civitas)
        """
        if initial_guess is None:
            initial_guess = [1.0, 1.0]
        
        # Use scipy's optimization to find the best coefficients
        result = optimize.minimize(
            lambda x: self.objective_function(x, observed_data),
            initial_guess,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'disp': True}
        )
        
        self.coefficients = result.x
        print(f"Optimized coefficients: V_urbs = {self.coefficients[0]:.4f}, "
              f"V_civitas = {self.coefficients[1]:.4f}")
        return self.coefficients
    
    def calculate_energy(self):
        """
        Calculate the energy based on the optimized potential field.
        
        Returns:
        --------
        numpy.ndarray
            The calculated energy field
        """
        if self.coefficients is None:
            raise ValueError("Coefficients must be optimized first")
        
        self.potential_field = self.calculate_potential_field(
            self.coefficients[0], self.coefficients[1]
        )
        
        # Energy is negative of potential
        self.energy = -self.potential_field
        return self.energy
    
    def apply_boltzmann_distribution(self):
        """
        Apply Boltzmann distribution to calculate probability.
        
        Returns:
        --------
        numpy.ndarray
            The probability distribution
        """
        if self.energy is None:
            self.calculate_energy()
        
        # Use mean of absolute energy as temperature parameter to normalize distribution
        temperature = np.mean(np.abs(self.energy))
        boltzmann = np.exp(-self.energy / temperature)
        
        # Normalize to get probability distribution
        self.probability_map = boltzmann / np.sum(boltzmann)
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
        if self.probability_map is None:
            self.apply_boltzmann_distribution()
        
        # Apply power transformation to enhance contrast
        enhanced = self.probability_map ** enhancement_factor
        
        # Re-normalize
        self.enhanced_probability = enhanced / np.sum(enhanced)
        return self.enhanced_probability
    
    def validate(self, observed_data):
        """
        Calculate validation metrics for the model.
        
        Parameters:
        -----------
        observed_data : numpy.ndarray
            Observed pedestrian density for validation
        
        Returns:
        --------
        dict
            Dictionary containing validation metrics
        """
        if self.enhanced_probability is None:
            self.enhance_distribution()
        
        # Ensure observed data has the same shape
        if observed_data.shape != self.enhanced_probability.shape:
            raise ValueError("Observed data must have the same shape as model outputs")
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(observed_data, self.enhanced_probability))
        
        # Calculate Pearson correlation
        # Flatten arrays for correlation calculation
        flat_observed = observed_data.flatten()
        flat_predicted = self.enhanced_probability.flatten()
        
        # Filter out NaN values
        valid_indices = ~(np.isnan(flat_observed) | np.isnan(flat_predicted))
        correlation, p_value = pearsonr(
            flat_observed[valid_indices], 
            flat_predicted[valid_indices]
        )
        
        validation_results = {
            'RMSE': rmse,
            'Pearson_r': correlation,
            'p_value': p_value
        }
        
        print(f"Validation results:")
        print(f"RMSE: {rmse:.6f}")
        print(f"Spatial correlation (Pearson r): {correlation:.6f} (p-value: {p_value:.6f})")
        
        return validation_results
    
    def save_outputs(self):
        """
        Save the model outputs to files.
        """
        if self.enhanced_probability is None:
            self.enhance_distribution()
        
        # Save probability map as raster
        output_path = os.path.join(self.output_dir, 'pedestrian_probability.tif')
        
        # Update metadata for output
        out_meta = self.metadata.copy()
        out_meta.update({"driver": "GTiff", "dtype": "float32"})
        
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write(self.enhanced_probability.astype(np.float32), 1)
        
        # Save visualization as PNG
        plt.figure(figsize=(12, 8))
        plt.imshow(self.enhanced_probability, cmap='hot')
        plt.colorbar(label='Pedestrian Probability')
        plt.title('Pedestrian Probability Map (Classical Model)')
        plt.savefig(os.path.join(self.output_dir, 'pedestrian_probability.png'), dpi=300)
        
        print(f"Outputs saved to {self.output_dir}")
    
    def run_model(self, v_urbs_path, v_civitas_path, observed_data=None, 
                  enhancement_factor=1.5, save_results=True):
        """
        Run the complete classical pedestrian probability model.
        
        Parameters:
        -----------
        v_urbs_path : str
            Path to V urbs raster file
        v_civitas_path : str
            Path to V civitas raster file
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
        # Load data
        self.load_rasters(v_urbs_path, v_civitas_path)
        
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
        
        return {
            'coefficients': self.coefficients,
            'probability_map': self.enhanced_probability,
            'validation': validation_metrics
        }


def main():
    """
    Example usage of the ClassicalPedestrianModel.
    """
    # Create model instance
    model = ClassicalPedestrianModel(output_dir='./outputs')
    
    # Example paths - update with your actual file paths
    v_urbs_path = 'data/v_urbs.tif'
    v_civitas_path = 'data/v_civitas.tif'
    
    # If you have observed data for validation:
    # observed_data = rasterio.open('data/observed_pedestrian.tif').read(1)
    # results = model.run_model(v_urbs_path, v_civitas_path, observed_data)
    
    # If you don't have observed data:
    results = model.run_model(v_urbs_path, v_civitas_path)
    
    print("Model execution completed.")


if __name__ == "__main__":
    main()