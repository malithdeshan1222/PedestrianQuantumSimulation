import os
import argparse
import logging
from geotiff_processor import GeotiffProcessor

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process urban factors from GeoTIFF files')
    
    # Required arguments for input files
    parser.add_argument('--input-dir', type=str, required=True,
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
    
    # Initialize GeoTIFF processor
    processor = GeotiffProcessor(args.output_dir)
    
    # Map of factor names to expected file patterns
    factor_file_patterns = {
        'building_density': ['building_density', 'buildingdensity', 'bldg_density'],
        'building_height': ['building_height', 'buildingheight', 'bldg_height'],
        'tree_height': ['tree_height', 'treeheight'],
        'wall_constraint': ['wall_constraint', 'wallconstraint'],
        'mean_depth': ['mean_depth', 'meandepth'],
        'street_centrality': ['street_centrality', 'streetcentrality'],
        'poi_density': ['poi_density', 'poidensity'],
        'pedestrian_accessibility': ['pedestrian_accessibility', 'pedestrianaccessibility', 'ped_access'],
        'isovist': ['isovist'],
        'vehicle_density': ['vehicle_density', 'vehicledensity'],
        'ground_truth': ['ground_truth', 'groundtruth', 'observed']
    }
    
    # Try to find and load all factor files
    for factor, patterns in factor_file_patterns.items():
        found = False
        for pattern in patterns:
            for filename in os.listdir(args.input_dir):
                if pattern.lower() in filename.lower() and filename.lower().endswith(('.tif', '.tiff')):
                    filepath = os.path.join(args.input_dir, filename)
                    if processor.load_geotiff(factor, filepath):
                        logger.info(f"Loaded {factor} from {filepath}")
                        found = True
                        break
            if found:
                break
        
        if not found:
            logger.warning(f"Could not find GeoTIFF for factor: {factor}")
    
    # Align all rasters to same spatial reference and dimensions
    processor.align_rasters()
    
    # Process the potential field
    upf, success = processor.process_potential_field(
        alpha=args.alpha,
        beta=args.beta,
        temperature=args.temperature
    )
    
    if success:
        logger.info("Processing completed successfully")
        
        # Print validation metrics if ground truth was provided
        if hasattr(upf, 'ground_truth') and upf.ground_truth is not None:
            try:
                metrics = upf.calculate_validation_metrics()
                logger.info("Validation metrics:")
                logger.info(f"  RMSE: {metrics['rmse']:.6f}")
                logger.info(f"  Pearson's r: {metrics['pearson_r']:.6f}")
                logger.info(f"  p-value: {metrics['p_value']:.6e}")
            except Exception as e:
                logger.error(f"Error calculating validation metrics: {str(e)}")
    else:
        logger.error("Processing failed")

if __name__ == "__main__":
    main()