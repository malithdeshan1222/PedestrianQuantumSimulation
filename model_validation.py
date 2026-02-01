import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from itertools import product

# Sensitivity analysis for model coefficients
def perform_sensitivity_analysis(base_model_fn, base_coeffs, parameter_ranges, observed_data):
    """
    Perform sensitivity analysis by varying coefficients and measuring impact
    
    Args:
        base_model_fn: Function that computes model output given coefficients
        base_coeffs: Base coefficients to perturb
        parameter_ranges: Dict of parameter indices and ranges to test
        observed_data: Ground truth for validation
    """
    results = {}
    base_performance = None
    
    # First, evaluate the base performance
    base_output = base_model_fn(base_coeffs)
    base_correlation, _ = pearsonr(base_output.flatten(), observed_data.flatten())
    base_performance = base_correlation
    
    # For each parameter to analyze
    for param_idx, range_vals in parameter_ranges.items():
        param_results = []
        
        for val in range_vals:
            # Create modified coefficients
            test_coeffs = base_coeffs.copy()
            test_coeffs[param_idx] = val
            
            # Run model with modified coefficients
            output = base_model_fn(test_coeffs)
            
            # Calculate correlation with observed data
            corr, _ = pearsonr(output.flatten(), observed_data.flatten())
            
            # Store result
            param_results.append((val, corr))
        
        results[param_idx] = param_results
    
    # Plot results
    fig, axes = plt.subplots(len(parameter_ranges), 1, figsize=(10, 4*len(parameter_ranges)))
    if len(parameter_ranges) == 1:
        axes = [axes]
        
    for i, (param_idx, param_results) in enumerate(results.items()):
        x_vals, y_vals = zip(*param_results)
        axes[i].plot(x_vals, y_vals, 'o-')
        axes[i].axhline(y=base_performance, color='r', linestyle='--', label='Base Performance')
        axes[i].set_xlabel(f'Coefficient {param_idx} Value')
        axes[i].set_ylabel('Correlation with Observed Data')
        axes[i].set_title(f'Sensitivity Analysis for Coefficient {param_idx}')
        axes[i].grid(True)
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('sensitivity_analysis.png', dpi=300)
    plt.show()
    
    return results

# Cross-validation for model performance
def perform_cross_validation(model_fn, data_regions, k_folds=4):
    """
    Perform spatial k-fold cross-validation
    
    Args:
        model_fn: Model function that takes training data and returns predictions
        data_regions: Dict with 'inputs' (features) and 'target' (ground truth)
        k_folds: Number of spatial folds
    """
    # Get data dimensions
    h, w = data_regions['target'].shape
    
    # Create spatial folds (divide the map into k×k regions)
    fold_h, fold_w = h // k_folds, w // k_folds
    
    results = []
    
    # For each fold as test set
    for i, j in product(range(k_folds), range(k_folds)):
        # Define test region
        test_mask = np.zeros((h, w), dtype=bool)
        test_mask[i*fold_h:(i+1)*fold_h, j*fold_w:(j+1)*fold_w] = True
        
        # Training mask is the inverse
        train_mask = ~test_mask
        
        # Extract training and test data
        train_data = {
            'inputs': [f[train_mask] for f in data_regions['inputs']],
            'target': data_regions['target'][train_mask]
        }
        
        test_data = {
            'inputs': [f[test_mask] for f in data_regions['inputs']],
            'target': data_regions['target'][test_mask]
        }
        
        # Train model
        model = model_fn(train_data)
        
        # Make predictions
        predictions = model(test_data['inputs'])
        
        # Evaluate
        corr, _ = pearsonr(predictions.flatten(), test_data['target'].flatten())
        
        results.append({
            'fold': (i, j),
            'correlation': corr,
            'test_mask': test_mask
        })
    
    # Calculate average performance
    avg_corr = np.mean([r['correlation'] for r in results])
    print(f"Average correlation across {k_folds}×{k_folds} spatial CV: {avg_corr:.4f}")
    
    return results, avg_corr