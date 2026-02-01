import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error
import os

class UrbanPotentialField:
    def __init__(self, grid_size=(100, 100)):
        """
        Initialize the Urban Potential Field calculator.
        
        Parameters:
        -----------
        grid_size : tuple
            Size of the grid (height, width)
        """
        self.grid_size = grid_size
        self.v_urbs = None
        self.v_civitas = None
        self.potential_field = None
        self.probability_dist = None
        self.ground_truth = None
        
    def set_v_urbs_factors(self, building_density, building_height, 
                          tree_height, wall_constraint, 
                          mean_depth, street_centrality):
        """
        Set the urban physical factors (V urbs).
        
        All inputs should be numpy arrays of shape grid_size.
        """
        # Normalize all factors to [0, 1] range if they're not already
        factors = [
            self._normalize(building_density),
            self._normalize(building_height),
            self._normalize(tree_height),
            self._normalize(wall_constraint),
            self._normalize(mean_depth),
            self._normalize(street_centrality)
        ]
        
        # Calculate V urbs as weighted sum of factors
        # Weights can be adjusted based on domain knowledge
        weights = np.array([0.25, 0.2, 0.15, 0.15, 0.1, 0.15])
        
        self.v_urbs_components = {
            'building_density': factors[0],
            'building_height': factors[1],
            'tree_height': factors[2],
            'wall_constraint': factors[3],
            'mean_depth': factors[4],
            'street_centrality': factors[5]
        }
        
        self.v_urbs = np.zeros(self.grid_size)
        for i, factor in enumerate(factors):
            self.v_urbs += weights[i] * factor
            
        return self.v_urbs
        
    def set_v_civitas_factors(self, poi_density, pedestrian_accessibility, 
                            isovist, vehicle_density):
        """
        Set the civic/social factors (V civitas).
        
        All inputs should be numpy arrays of shape grid_size.
        """
        # Normalize all factors to [0, 1] range if they're not already
        factors = [
            self._normalize(poi_density),
            self._normalize(pedestrian_accessibility),
            self._normalize(isovist),
            self._normalize(vehicle_density)
        ]
        
        # Calculate V civitas as weighted sum of factors
        # Weights can be adjusted based on domain knowledge
        weights = np.array([0.3, 0.3, 0.2, 0.2])
        
        self.v_civitas_components = {
            'poi_density': factors[0],
            'pedestrian_accessibility': factors[1],
            'isovist': factors[2],
            'vehicle_density': factors[3]
        }
        
        self.v_civitas = np.zeros(self.grid_size)
        for i, factor in enumerate(factors):
            self.v_civitas += weights[i] * factor
            
        return self.v_civitas
    
    def generate_potential_field(self, alpha=0.5, beta=0.5):
        """
        Generate the potential field by combining V urbs and V civitas.
        
        Parameters:
        -----------
        alpha : float
            Weight for V urbs
        beta : float
            Weight for V civitas
        """
        if self.v_urbs is None or self.v_civitas is None:
            raise ValueError("V urbs and V civitas must be set before generating potential field")
            
        self.potential_field = alpha * self.v_urbs + beta * self.v_civitas
        return self.potential_field
    
    def generate_probability_distribution(self, temperature=1.0):
        """
        Generate probability distribution using Boltzmann distribution.
        
        Parameters:
        -----------
        temperature : float
            Temperature parameter in the Boltzmann distribution
        """
        if self.potential_field is None:
            raise ValueError("Potential field must be generated first")
            
        # Apply Boltzmann distribution: P(s) ∝ exp(-E(s)/kT)
        # Where E(s) is the potential field, k is Boltzmann constant (set to 1),
        # and T is temperature
        energy = -self.potential_field  # Negative because we want higher potential to have higher probability
        boltzmann_factors = np.exp(energy / temperature)
        self.probability_dist = boltzmann_factors / np.sum(boltzmann_factors)  # Normalize
        
        return self.probability_dist
    
    def set_ground_truth(self, ground_truth):
        """
        Set the ground truth data for validation.
        
        Parameters:
        -----------
        ground_truth : numpy.ndarray
            Ground truth probability distribution
        """
        self.ground_truth = self._normalize(ground_truth)
        return self.ground_truth
    
    def calculate_validation_metrics(self):
        """
        Calculate validation metrics: RMSE and Pearson's r.
        """
        if self.probability_dist is None:
            raise ValueError("Probability distribution must be generated first")
            
        if self.ground_truth is None:
            raise ValueError("Ground truth must be set before calculating validation metrics")
            
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(self.ground_truth.flatten(), 
                                         self.probability_dist.flatten()))
        
        # Calculate Pearson's r
        pearson_r, p_value = stats.pearsonr(self.ground_truth.flatten(), 
                                           self.probability_dist.flatten())
        
        return {
            "rmse": rmse,
            "pearson_r": pearson_r,
            "p_value": p_value
        }
    
    def export_probability_map(self, filename="probability_map.png", dpi=300):
        """
        Export the probability map as an image.
        
        Parameters:
        -----------
        filename : str
            Output filename
        dpi : int
            Resolution of the output image
        """
        if self.probability_dist is None:
            raise ValueError("Probability distribution must be generated first")
        
        plt.figure(figsize=(10, 8))
        
        # Create a custom colormap from blue to red
        colors = [(0, 0, 0.8), (0, 0, 1), (0, 1, 1), (0, 1, 0), 
                 (1, 1, 0), (1, 0.5, 0), (1, 0, 0)]
        cmap_name = 'urban_potential'
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
        
        im = plt.imshow(self.probability_dist, cmap=cm)
        plt.colorbar(im, label='Probability')
        plt.title('Urban Activity Probability Map')
        plt.axis('off')
        
        # Save the figure
        plt.savefig(filename, bbox_inches='tight', dpi=dpi)
        plt.close()
        
        print(f"Probability map exported as {filename}")
        
        return filename
    
    def _normalize(self, data):
        """
        Normalize data to [0, 1] range.
        """
        data_min = np.min(data)
        data_max = np.max(data)
        
        if data_min == data_max:
            return np.zeros_like(data)
            
        return (data - data_min) / (data_max - data_min)
    
    def generate_summary_report(self, output_dir="./output"):
        """
        Generate a summary report of the analysis.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save outputs
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Export all maps
        potential_field_file = os.path.join(output_dir, "potential_field_map.png")
        self.export_map(self.potential_field, potential_field_file, "Potential Field")
        
        v_urbs_file = os.path.join(output_dir, "v_urbs_map.png")
        self.export_map(self.v_urbs, v_urbs_file, "Urban Physical Factors (V urbs)")
        
        v_civitas_file = os.path.join(output_dir, "v_civitas_map.png")
        self.export_map(self.v_civitas, v_civitas_file, "Civic/Social Factors (V civitas)")
        
        prob_file = os.path.join(output_dir, "probability_map.png")
        self.export_map(self.probability_dist, prob_file, "Probability Distribution", cmap='viridis')
        
        # Calculate validation metrics if ground truth is available
        metrics = {}
        if self.ground_truth is not None:
            metrics = self.calculate_validation_metrics()
            ground_truth_file = os.path.join(output_dir, "ground_truth_map.png")
            self.export_map(self.ground_truth, ground_truth_file, "Ground Truth")
            
            diff_map = np.abs(self.probability_dist - self.ground_truth)
            diff_file = os.path.join(output_dir, "difference_map.png")
            self.export_map(diff_map, diff_file, "Absolute Difference", cmap='hot')
        
        # Generate summary text file
        summary_file = os.path.join(output_dir, "analysis_summary.txt")
        with open(summary_file, "w") as f:
            f.write("Urban Potential Field Analysis Summary\n")
            f.write("=====================================\n\n")
            
            f.write("Grid Size: {}\n\n".format(self.grid_size))
            
            f.write("Potential Field Statistics:\n")
            f.write("  Min: {:.4f}\n".format(np.min(self.potential_field)))
            f.write("  Max: {:.4f}\n".format(np.max(self.potential_field)))
            f.write("  Mean: {:.4f}\n".format(np.mean(self.potential_field)))
            f.write("  Std Dev: {:.4f}\n\n".format(np.std(self.potential_field)))
            
            f.write("Probability Distribution Statistics:\n")
            f.write("  Min: {:.6f}\n".format(np.min(self.probability_dist)))
            f.write("  Max: {:.6f}\n".format(np.max(self.probability_dist)))
            f.write("  Mean: {:.6f}\n".format(np.mean(self.probability_dist)))
            f.write("  Std Dev: {:.6f}\n\n".format(np.std(self.probability_dist)))
            
            if metrics:
                f.write("Validation Metrics:\n")
                f.write("  RMSE: {:.6f}\n".format(metrics["rmse"]))
                f.write("  Pearson's r: {:.6f}\n".format(metrics["pearson_r"]))
                f.write("  p-value: {:.6e}\n\n".format(metrics["p_value"]))
                
            f.write("Files Generated:\n")
            f.write("  - {}\n".format(potential_field_file))
            f.write("  - {}\n".format(v_urbs_file))
            f.write("  - {}\n".format(v_civitas_file))
            f.write("  - {}\n".format(prob_file))
            
            if self.ground_truth is not None:
                f.write("  - {}\n".format(ground_truth_file))
                f.write("  - {}\n".format(diff_file))
                
        print(f"Summary report generated at {summary_file}")
        return summary_file
        
    def export_map(self, data, filename, title, cmap='viridis'):
        """
        Export a map as an image.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Data to visualize
        filename : str
            Output filename
        title : str
            Title for the plot
        cmap : str
            Matplotlib colormap name
        """
        plt.figure(figsize=(10, 8))
        im = plt.imshow(data, cmap=cmap)
        plt.colorbar(im)
        plt.title(title)
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Map exported as {filename}")