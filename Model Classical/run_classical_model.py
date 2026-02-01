#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to run the classical pedestrian model with command line options.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Ensure that the project root is in the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from classical_pedestrian_model import ClassicalPedestrianModel, optimized_load_geotiff


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Classical Pedestrian Model')
    parser.add_argument('--data_path', type=str, default='data',
                        help='Path to directory containing raster data')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs')
    parser.add_argument('--enhancement', type=float, default=1.5,
                        help='Enhancement factor for probability distribution')
    parser.add_argument('--optimization_method', type=str, default='Nelder-Mead', 
                        choices=['Nelder-Mead', 'BFGS', 'Powell', 'SLSQP', 'CG'],
                        help='Optimization method for coefficient estimation')
    parser.add_argument('--plot_factors', action='store_true',
                        help='Generate plots of all input factors')
    
    return parser.parse_args()


def plot_factors(factors, output_dir):
    """
    Generate plots of all input factors for visualization.
    
    Parameters:
    -----------
    factors : dict
        Dictionary of factor arrays
    output_dir : str
        Output directory for saving plots
    """
    factors_dir = os.path.join(output_dir, 'factors')
    os.makedirs(factors_dir, exist_ok=True)
    
    # Determine grid size for combined plot
    n_factors = len(factors)
    ncols = min(3, n_factors)
    nrows = (n_factors + ncols - 1) // ncols
    
    # Create individual plots and a combined plot
    plt.figure(figsize=(ncols * 5, nrows * 4))
    
    for i, (name, data) in enumerate(factors.items()):
        # Create individual plot
        plt.figure(figsize=(8, 6))
        plt.imshow(data, cmap='viridis')
        plt.colorbar(label=name)
        plt.title(name.replace('_', ' ').title())
        plt.tight_layout()
        plt.savefig(os.path.join(factors_dir, f"{name}.png"), dpi=200)
        plt.close()
        
        # Add to combined plot
        plt.figure(1)
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(data, cmap='viridis')
        plt.title(name.replace('_', ' ').title())
        plt.axis('off')
    
    plt.figure(1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_factors.png'), dpi=300)
    plt.close()


def main():
    """Run the classical pedestrian model."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Running Classical Pedestrian Model")
    print(f"----------------------------------")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Enhancement factor: {args.enhancement}")
    print(f"Optimization method: {args.optimization_method}")
    
    # Create model instance
    model = ClassicalPedestrianModel(output_dir=args.output_dir)
    
    # Load factors
    model.load_factors(data_path=args.data_path)
    
    # Plot factors if requested
    if args.plot_factors:
        print("Generating factor visualizations...")
        plot_factors(model.factors, args.output_dir)
    
    # Normalize factors
    model.normalize_factors()
    
    # Run optimization
    model.optimize_coefficients(
        observed_data=model.pedestrian_density,
        method=args.optimization_method
    )
    
    # Calculate energy and probability
    model.calculate_energy()
    model.apply_boltzmann_distribution()
    model.enhance_distribution(enhancement_factor=args.enhancement)
    
    # Validate results
    validation = model.validate()
    
    # Save outputs
    model.save_outputs()
    
    # Print summary
    print("\nModel Results Summary:")
    print("-----------------------")
    print("Optimized Factor Coefficients:")
    for factor, coef in zip(model.factors.keys(), model.coefficients):
        print(f"  {factor}: {coef:.6f}")
    
    if validation:
        print("\nValidation Metrics:")
        print(f"  RMSE: {validation['RMSE']:.6f}")
        print(f"  Spatial correlation (Pearson r): {validation['Pearson_r']:.6f}")
        print(f"  p-value: {validation['p_value']:.6f}")
    
    print(f"\nOutputs saved to {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()