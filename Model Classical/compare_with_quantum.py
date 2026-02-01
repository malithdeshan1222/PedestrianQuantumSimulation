#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compare results of classical pedestrian model with quantum model.
This only works if you've already run both models.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


def load_raster(file_path):
    """Load a GeoTIFF raster file."""
    try:
        with rasterio.open(file_path) as src:
            return src.read(1), src.meta
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None


def compare_models(classical_dir, quantum_dir, output_dir, observed_data_path=None):
    """
    Compare the results of classical and quantum pedestrian models.
    
    Parameters:
    -----------
    classical_dir : str
        Directory containing classical model results
    quantum_dir : str
        Directory containing quantum model results
    output_dir : str
        Directory to save comparison results
    observed_data_path : str, optional
        Path to observed pedestrian density file for validation
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model outputs
    classical_prob, _ = load_raster(os.path.join(classical_dir, 'pedestrian_probability.tif'))
    quantum_prob, _ = load_raster(os.path.join(quantum_dir, 'pedestrian_probability.tif'))
    
    if classical_prob is None or quantum_prob is None:
        print("Error: Could not load model results")
        return
    
    # Ensure both have same shape
    if classical_prob.shape != quantum_prob.shape:
        print(f"Error: Shape mismatch - Classical: {classical_prob.shape}, Quantum: {quantum_prob.shape}")
        return
    
    # Calculate difference
    difference = quantum_prob - classical_prob
    
    # Load observed data if provided
    observed_data = None
    if observed_data_path:
        observed_data, _ = load_raster(observed_data_path)
        if observed_data is None:
            print(f"Warning: Could not load observed data from {observed_data_path}")
    
    # Create comparison visualization
    fig, axs = plt.subplots(2, 2, figsize=(16, 14))
    
    # Classical model output
    im1 = axs[0, 0].imshow(classical_prob, cmap='hot')
    axs[0, 0].set_title('Classical Model Prediction')
    fig.colorbar(im1, ax=axs[0, 0], shrink=0.7)
    
    # Quantum model output
    im2 = axs[0, 1].imshow(quantum_prob, cmap='hot')
    axs[0, 1].set_title('Quantum Model Prediction')
    fig.colorbar(im2, ax=axs[0, 1], shrink=0.7)
    
    # Difference between models
    im3 = axs[1, 0].imshow(difference, cmap='coolwarm', vmin=-0.05, vmax=0.05)
    axs[1, 0].set_title('Difference (Quantum - Classical)')
    fig.colorbar(im3, ax=axs[1, 0], shrink=0.7)
    
    # Observed data or correlation plot
    if observed_data is not None:
        if observed_data.shape == classical_prob.shape:
            im4 = axs[1, 1].imshow(observed_data, cmap='viridis')
            axs[1, 1].set_title('Observed Pedestrian Density')
            fig.colorbar(im4, ax=axs[1, 1], shrink=0.7)
            
            # Calculate validation metrics
            # Create mask for valid data
            mask = ~(np.isnan(observed_data) | np.isnan(classical_prob) | np.isnan(quantum_prob))
            
            # Classical model metrics
            rmse_classical = np.sqrt(mean_squared_error(
                observed_data[mask], classical_prob[mask]
            ))
            corr_classical, p_classical = pearsonr(
                observed_data[mask].flatten(), classical_prob[mask].flatten()
            )
            
            # Quantum model metrics
            rmse_quantum = np.sqrt(mean_squared_error(
                observed_data[mask], quantum_prob[mask]
            ))
            corr_quantum, p_quantum = pearsonr(
                observed_data[mask].flatten(), quantum_prob[mask].flatten()
            )
            
            # Add metrics as text
            metrics_text = (
                f"Classical Model:\n"
                f"  RMSE: {rmse_classical:.6f}\n"
                f"  Pearson r: {corr_classical:.6f}\n\n"
                f"Quantum Model:\n"
                f"  RMSE: {rmse_quantum:.6f}\n"
                f"  Pearson r: {corr_quantum:.6f}\n\n"
                f"Improvement:\n"
                f"  RMSE: {(rmse_classical-rmse_quantum)/rmse_classical*100:.2f}%\n"
                f"  Corr: {(corr_quantum-corr_classical)/abs(corr_classical)*100:.2f}%"
            )
            
            # Save metrics to file
            with open(os.path.join(output_dir, 'comparison_metrics.txt'), 'w') as f:
                f.write(metrics_text)
        else:
            axs[1, 1].text(0.5, 0.5, "Observed data shape mismatch",
                         horizontalalignment='center', verticalalignment='center')
            axs[1, 1].set_title('Error')
    else:
        # Calculate correlation between models
        mask = ~(np.isnan(classical_prob) | np.isnan(quantum_prob))
        corr, _ = pearsonr(classical_prob[mask].flatten(), quantum_prob[mask].flatten())
        
        # Scatter plot of quantum vs classical
        axs[1, 1].scatter(classical_prob[mask].flatten(), quantum_prob[mask].flatten(), 
                        alpha=0.5, s=1, c='blue')
        axs[1, 1].plot([0, max(np.nanmax(classical_prob), np.nanmax(quantum_prob))], 
                      [0, max(np.nanmax(classical_prob), np.nanmax(quantum_prob))], 
                      'r--')
        axs[1, 1].set_xlabel('Classical Model Probability')
        axs[1, 1].set_ylabel('Quantum Model Probability')
        axs[1, 1].set_title(f'Model Correlation (r = {corr:.4f})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
    plt.close()
    
    print(f"Comparison completed and saved to {output_dir}")


def main():
    """Parse arguments and run comparison."""
    parser = argparse.ArgumentParser(description='Compare Classical and Quantum Pedestrian Models')
    parser.add_argument('--classical_dir', type=str, default='outputs',
                        help='Directory containing classical model results')
    parser.add_argument('--quantum_dir', type=str, default='quantum_outputs',
                        help='Directory containing quantum model results')
    parser.add_argument('--output_dir', type=str, default='comparison',
                        help='Directory to save comparison results')
    parser.add_argument('--observed_data', type=str, default=None,
                        help='Path to observed pedestrian density file')
    
    args = parser.parse_args()
    
    compare_models(
        args.classical_dir,
        args.quantum_dir,
        args.output_dir,
        args.observed_data
    )


if __name__ == "__main__":
    main()