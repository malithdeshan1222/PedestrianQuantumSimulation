import os
import argparse
import logging
from geotiff_processor import GeotiffProcessor

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process urban factors from GeoTIFF files')
    
    # Set your specific path as the default
    default_input_dir = r"D:\DATA\MALITH\Uni\Semester 08\ISRP\Model GIS\Model_VS_Qiskit\Model Classical\data"
    
    # Required arguments for input files (with your path as default)
    parser.add_argument('--input-dir', type=str, default=default_input_dir,
                      help='Directory containing input GeoTIFF files')
    parser.add_argument('--output-dir', type=str, default='./output',
                      help='Directory to save output files')
    
    # Optional parameters for the model
    parser.add_argument('--alpha', type=float, default=0.5,
                      help='Weight for V urbs (default: 0.5)')
    parser.add_argument('--beta', type=float, default=0.5,
                      help='Weight for V civitas (default: 0.5)')
    parser.add_argument('--temperature', type=float, default=0.1,
                      help='Temperature parameter for Boltzmann distribution (default: 0.1)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return
    
    # Rest of the code remains the same
    # ...

# Rest of the file remains unchanged