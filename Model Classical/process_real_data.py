import numpy as np
import pandas as pd
import os
from urban_potential_field import UrbanPotentialField
import matplotlib.pyplot as plt

def load_data_from_csv(filename, grid_size=(100, 100)):
    """
    Load data from CSV file and reshape to the specified grid size.
    
    CSV should contain columns for each factor, with x and y coordinates.
    """
    try:
        df = pd.read_csv(filename)
        
        # Extract coordinates
        x_coords = df['x'].values
        y_coords = df['y'].values
        
        # Create empty grids for each factor
        factor_grids = {}
        
        # Extract known factor columns from dataframe
        factor_columns = [
            'building_density', 'building_height', 'tree_height', 
            'wall_constraint', 'mean_depth', 'street_centrality',
            'poi_density', 'pedestrian_accessibility', 'isovist', 
            'vehicle_density', 'ground_truth'
        ]
        
        # Keep only columns that exist in the dataframe
        factor_columns = [col for col in factor_columns if col in df.columns]
        
        # For each factor, create a grid
        for factor in factor_columns:
            if factor in df.columns:
                grid = np.zeros(grid_size)
                values = df[factor].values
                
                # Scale x, y coordinates to grid indices
                x_indices = np.clip((x_coords / np.max(x_coords) * (grid_size[1]-1)).astype(int), 0, grid_size[1]-1)
                y_indices = np.clip((y_coords / np.max(y_coords) * (grid_size[0]-1)).astype(int), 0, grid_size[0]-1)
                
                # Fill grid with values
                for i in range(len(values)):
                    grid[y_indices[i], x_indices[i]] = values[i]
                
                factor_grids[factor] = grid
        
        return factor_grids
    
    except Exception as e:
        print(f"Error loading data from {filename}: {str(e)}")
        return None

def main():
    # File path to your CSV data
    csv_file = "urban_data.csv"  # Replace with your actual file path
    grid_size = (100, 100)  # Adjust based on your needs
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        print("Using synthetic data instead...")
        
        # If file doesn't exist, use synthetic data as a fallback
        from example_usage import create_synthetic_data
        data_dict = create_synthetic_data(grid_size)
    else:
        # Load data from CSV
        data_dict = load_data_from_csv(csv_file, grid_size)
        
        if data_dict is None:
            print("Error loading data. Using synthetic data instead...")
            from example_usage import create_synthetic_data
            data_dict = create_synthetic_data(grid_size)
        else:
            # Reorganize data dictionary to match expected structure
            data_dict = {
                'v_urbs': {
                    'building_density': data_dict.get('building_density', np.zeros(grid_size)),
                    'building_height': data_dict.get('building_height', np.zeros(grid_size)),
                    'tree_height': data_dict.get('tree_height', np.zeros(grid_size)),
                    'wall_constraint': data_dict.get('wall_constraint', np.zeros(grid_size)),
                    'mean_depth': data_dict.get('mean_depth', np.zeros(grid_size)),
                    'street_centrality': data_dict.get('street_centrality', np.zeros(grid_size))
                },
                'v_civitas': {
                    'poi_density': data_dict.get('poi_density', np.zeros(grid_size)),
                    'pedestrian_accessibility': data_dict.get('pedestrian_accessibility', np.zeros(grid_size)),
                    'isovist': data_dict.get('isovist', np.zeros(grid_size)),
                    'vehicle_density': data_dict.get('vehicle_density', np.zeros(grid_size))
                },
                'ground_truth': data_dict.get('ground_truth', None)
            }
    
    # Initialize Urban Potential Field calculator
    upf = UrbanPotentialField(grid_size)
    
    # Set V urbs factors
    upf.set_v_urbs_factors(
        data_dict['v_urbs']['building_density'],
        data_dict['v_urbs']['building_height'],
        data_dict['v_urbs']['tree_height'],
        data_dict['v_urbs']['wall_constraint'],
        data_dict['v_urbs']['mean_depth'],
        data_dict['v_urbs']['street_centrality']
    )
    
    # Set V civitas factors
    upf.set_v_civitas_factors(
        data_dict['v_civitas']['poi_density'],
        data_dict['v_civitas']['pedestrian_accessibility'],
        data_dict['v_civitas']['isovist'],
        data_dict['v_civitas']['vehicle_density']
    )
    
    # Generate potential field
    # You can adjust alpha and beta to change the relative importance of V urbs and V civitas
    potential_field = upf.generate_potential_field(alpha=0.5, beta=0.5)
    print("Potential field generated")
    
    # Generate probability distribution using Boltzmann distribution
    # You can adjust temperature to control the "sharpness" of the distribution
    prob_dist = upf.generate_probability_distribution(temperature=0.1)
    print("Probability distribution generated")
    
    # Set ground truth for validation if available
    if data_dict.get('ground_truth') is not None:
        upf.set_ground_truth(data_dict['ground_truth'])
        
        # Calculate validation metrics
        metrics = upf.calculate_validation_metrics()
        print("Validation metrics:")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  Pearson's r: {metrics['pearson_r']:.6f}")
        print(f"  p-value: {metrics['p_value']:.6e}")
    
    # Export maps and generate report
    upf.generate_summary_report("./output")

if __name__ == "__main__":
    main()