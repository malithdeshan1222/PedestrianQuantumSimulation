#!/usr/bin/env python3
"""
Quantum Urban Planning Model
============================================
A quantum computing approach to urban pedestrian movement simulation.

This model simulates pedestrian movement patterns in urban environments
using quantum mechanical principles to solve the urban potential field.
"""
import matplotlib
matplotlib.use('Agg') 
import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

# Import core model components
from config import Config, ModelConfig, QuantumConfig, DataConfig
from enhanced_model import run_enhanced_quantum_urban_model, delay_job
import time

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Quantum Urban Planning Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--data-path', type=str,
        help='Path to input data directory'
    )
    
    parser.add_argument(
        '--result-path', type=str,
        help='Path to save results'
    )
    
    parser.add_argument(
        '--no-quantum', action='store_true',
        help='Disable quantum computing (use classical approach)'
    )
    
    parser.add_argument(
        '--real-device', action='store_true',
        help='Use real quantum device instead of simulator'
    )
    
    parser.add_argument(
        '--api-token', type=str,
        help='IBM Quantum API token'
    )
    
    parser.add_argument(
        '--time-periods', type=str, nargs='+',
        choices=['morning', 'afternoon', 'evening', 'night'],
        default=['morning', 'afternoon', 'evening'],
        help='Time periods to simulate'
    )
    
    parser.add_argument(
        '--quantum-coeffs', action='store_true',
        help='Use quantum computing to optimize coefficients'
    )
    
    return parser.parse_args()

def validate_config(config):
    """Validate and ensure all required attributes are present."""
    # Ensure model config has required attributes
    if not hasattr(config.model, 'use_quantum_coeffs'):
        config.model.use_quantum_coeffs = False
        print("Warning: Added missing 'use_quantum_coeffs' attribute (default: False)")
        
    if not hasattr(config.model, 'learn_coeffs_from_data'):
        config.model.learn_coeffs_from_data = True
        print("Warning: Added missing 'learn_coeffs_from_data' attribute (default: True)")
        
    # Print configuration info for debugging
    print("Configuration Structure:")
    print(f"Type of config: {type(config)}")
    print(f"Type of config.model: {type(config.model)}")
    print(f"Model attributes: {[attr for attr in dir(config.model) if not attr.startswith('_')]}")
    
    return config

def main():
    """Main entry point for the quantum urban model."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Check for environment variable if API token not provided
    api_token = args.api_token
    if not api_token and 'IBMQ_API_TOKEN' in os.environ:
        api_token = os.environ['IBMQ_API_TOKEN']
    
    if not api_token and args.real_device:
        print("Warning: IBMQ_API_TOKEN environment variable not set.")
        print("To use a real quantum device, set this variable or provide the token with the --api-token option.")
    
    # Load configuration file if it exists
    try:
        config = Config.from_yaml(args.config)
        print(f"Config file {args.config} loaded successfully")
    except Exception as e:
        print(f"Config file {args.config} not found or invalid, using defaults: {str(e)}")
        config = Config()
    
    # Ensure config has all required attributes
    config = validate_config(config)
    
    # Override config with command line arguments
    if args.data_path:
        config.data.data_path = args.data_path
    
    if args.result_path:
        config.data.result_path = args.result_path
    
    if args.no_quantum:
        config.quantum.use_quantum = False
    
    if args.real_device:
        config.quantum.use_real_device = True
    
    if api_token:
        config.quantum.api_token = api_token
    
    if args.quantum_coeffs:
        config.model.use_quantum_coeffs = True
    
    # Add job delay configuration
    config.quantum.use_job_delay = True
    config.quantum.job_delay_seconds = 120  # 2 minutes
    
    # Print configuration summary
    print("Quantum Urban Planning Model")
    print("===========================")
    print(f"Using quantum computing: {config.quantum.use_quantum}")
    print(f"Using real quantum device: {config.quantum.use_real_device}")
    print(f"Using quantum coefficient optimization: {config.model.use_quantum_coeffs}")
    print(f"Using job delay (2 minutes): {config.quantum.use_job_delay}")
    print(f"Data path: {config.data.data_path}")
    print(f"Results will be saved to: {config.data.result_path}")
    print()
    
    # If no API token provided but real device requested, ask for it
    if config.quantum.use_real_device and not config.quantum.api_token:
        try:
            import getpass
            config.quantum.api_token = getpass.getpass("Enter your IBM Quantum API token: ")
        except Exception as e:
            print(f"Error getting API token: {e}, falling back to simulator")
            config.quantum.use_real_device = False
    
    # Restrict time periods to "morning" only
    time_periods = ["morning"]
    
    try:
        # Call the enhanced model with the delay function
        run_enhanced_quantum_urban_model(config, time_periods, delay_job)
    except Exception as e:
        print(f"Error running model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()