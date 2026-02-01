import numpy as np
from urban_potential_field import UrbanPotentialField

def create_synthetic_data(grid_size=(100, 100)):
    """
    Create synthetic data for demonstration.
    In a real scenario, this would be replaced with actual urban data.
    """
    # Create synthetic urban factors
    x, y = np.meshgrid(np.linspace(0, 1, grid_size[1]), np.linspace(0, 1, grid_size[0]))
    
    # V urbs factors
    building_density = np.exp(-((x-0.3)**2 + (y-0.7)**2)/0.1)
    building_height = np.exp(-((x-0.7)**2 + (y-0.2)**2)/0.2)
    tree_height = np.sin(10*x) * np.cos(10*y) * 0.5 + 0.5
    wall_constraint = np.exp(-((x-0.5)**2 + (y-0.5)**2)/0.3)
    mean_depth = np.exp(-((x-0.2)**2 + (y-0.8)**2)/0.15)
    street_centrality = np.abs(x-0.5) + np.abs(y-0.5)
    
    # V civitas factors
    poi_density = np.exp(-((x-0.6)**2 + (y-0.6)**2)/0.1)
    pedestrian_accessibility = 1 - np.exp(-((x-0.4)**2 + (y-0.3)**2)/0.2)
    isovist = np.exp(-((x-0.7)**2 + (y-0.7)**2)/0.25)
    vehicle_density = np.sin(5*x) * np.cos(5*y) * 0.5 + 0.5
    
    # Create synthetic ground truth for validation
    # This would be actual observed data in a real scenario
    ground_truth = np.exp(-((x-0.45)**2 + (y-0.55)**2)/0.15) * 0.8 + \
                  np.exp(-((x-0.6)**2 + (y-0.3)**2)/0.1) * 0.6
    ground_truth /= np.sum(ground_truth)  # Normalize
    
    return {
        'v_urbs': {
            'building_density': building_density,
            'building_height': building_height,
            'tree_height': tree_height,
            'wall_constraint': wall_constraint,
            'mean_depth': mean_depth,
            'street_centrality': street_centrality
        },
        'v_civitas': {
            'poi_density': poi_density,
            'pedestrian_accessibility': pedestrian_accessibility,
            'isovist': isovist,
            'vehicle_density': vehicle_density
        },
        'ground_truth': ground_truth
    }

def main():
    # Create grid size
    grid_size = (100, 100)
    
    # Create synthetic data (replace with real data in actual usage)
    data = create_synthetic_data(grid_size)
    
    # Initialize Urban Potential Field calculator
    upf = UrbanPotentialField(grid_size)
    
    # Set V urbs factors
    upf.set_v_urbs_factors(
        data['v_urbs']['building_density'],
        data['v_urbs']['building_height'],
        data['v_urbs']['tree_height'],
        data['v_urbs']['wall_constraint'],
        data['v_urbs']['mean_depth'],
        data['v_urbs']['street_centrality']
    )
    
    # Set V civitas factors
    upf.set_v_civitas_factors(
        data['v_civitas']['poi_density'],
        data['v_civitas']['pedestrian_accessibility'],
        data['v_civitas']['isovist'],
        data['v_civitas']['vehicle_density']
    )
    
    # Generate potential field
    # You can adjust alpha and beta to change the relative importance of V urbs and V civitas
    potential_field = upf.generate_potential_field(alpha=0.6, beta=0.4)
    print("Potential field generated")
    
    # Generate probability distribution
    # You can adjust temperature to control the "sharpness" of the distribution
    prob_dist = upf.generate_probability_distribution(temperature=0.1)
    print("Probability distribution generated")
    
    # Set ground truth for validation
    upf.set_ground_truth(data['ground_truth'])
    
    # Calculate validation metrics
    metrics = upf.calculate_validation_metrics()
    print("Validation metrics:")
    print("  RMSE:", metrics["rmse"])
    print("  Pearson's r:", metrics["pearson_r"])
    print("  p-value:", metrics["p_value"])
    
    # Export probability map
    upf.export_probability_map("probability_map.png")
    
    # Generate comprehensive report
    upf.generate_summary_report("./output")

if __name__ == "__main__":
    main()