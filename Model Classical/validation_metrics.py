import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error

def calculate_validation_metrics(predicted, observed):
    """
    Calculate validation metrics between predicted and observed values.
    
    Parameters:
    -----------
    predicted : numpy.ndarray
        Predicted values (probability distribution)
    observed : numpy.ndarray
        Observed values (ground truth)
        
    Returns:
    --------
    dict
        Dictionary containing validation metrics
    """
    # Flatten arrays
    pred_flat = predicted.flatten()
    obs_flat = observed.flatten()
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(obs_flat, pred_flat))
    
    # Calculate Pearson's r
    pearson_r, p_value = stats.pearsonr(obs_flat, pred_flat)
    
    return {
        "rmse": rmse,
        "pearson_r": pearson_r,
        "p_value": p_value
    }