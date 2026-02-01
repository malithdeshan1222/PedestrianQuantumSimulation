import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import warnings
import time
warnings.filterwarnings('ignore')

# Import our modules
from config import Config
from optimization_utils import (
    optimized_load_geotiff, batch_shadow_calculation, 
    apply_coefficients_vectorized, calculate_shadow_pedestrian_attraction, 
    learn_coefficients, save_geotiff
)
from quantum_improvements import solve_schrodinger_quantum, solve_classical_boltzmann  # Updated import
from model_enhancements import visualize_raster, advanced_visualization, create_pedestrian_flow_animation, ensure_same_shape
from quantum_coefficient_optimizer import QuantumCoefficientOptimizer
from scipy.ndimage import gaussian_filter

# Add this delay function to your script
def delay_job(delay_seconds=120):
    """
    Adds a delay before executing the next quantum job.
    
    Args:
        delay_seconds (int): Delay time in seconds, defaults to 120 seconds (2 minutes)
    """
    print(f"Adding delay of {delay_seconds} seconds before next job...")
    time.sleep(delay_seconds)
    print("Delay complete, continuing with next job.")

# Add this function after imports and before other functions
def enhance_patterns(probability_map, enhancement_factor=2.0):
    """Make pedestrian patterns more visible."""
    # Extract high-probability areas
    mean_prob = np.mean(probability_map)
    std_prob = np.std(probability_map)
    threshold = mean_prob + std_prob
    
    # Enhance contrast
    enhanced = np.zeros_like(probability_map)
    enhanced[probability_map > threshold] = probability_map[probability_map > threshold] * enhancement_factor
    enhanced[probability_map <= threshold] = probability_map[probability_map <= threshold] / enhancement_factor
    
    # Normalize
    enhanced = enhanced / np.sum(enhanced)
    
    return enhanced
    
def run_enhanced_quantum_urban_model(config=None, time_periods=None, job_delay_func=None):
    """Enhanced version of the quantum urban model with all improvements"""
    if config is None:
        config = Config()
        
    # If no job_delay_func is provided, use our default implementation
    if job_delay_func is None:
        job_delay_func = delay_job
        
    # Restrict time periods to "morning" only
    time_periods = ["morning"]
    
    print("Starting Enhanced Quantum Urban Model...")
    
    # Create results directory if it doesn't exist
    os.makedirs(config.data.result_path, exist_ok=True)
    
    # Load urban feature layers with optimized loading
    print("Loading urban data layers...")
    building_density = optimized_load_geotiff("building_density.tif", data_path=config.data.data_path)
    street_centrality = optimized_load_geotiff("street_centrality.tif", data_path=config.data.data_path)
    building_height = optimized_load_geotiff("building_height.tif", data_path=config.data.data_path)
    tree_height = optimized_load_geotiff("tree_height.tif", data_path=config.data.data_path)
    pedestrian_density = optimized_load_geotiff("pedestrian_density.tif", data_path=config.data.data_path)
    poi_density = optimized_load_geotiff("poi_density.tif", data_path=config.data.data_path)
    wall_constraint = optimized_load_geotiff("wall_constraint.tif", data_path=config.data.data_path)
    mean_depth = optimized_load_geotiff("mean_depth.tif", data_path=config.data.data_path)
    isovist = optimized_load_geotiff("isovist.tif", data_path=config.data.data_path)
    vehicle_density = optimized_load_geotiff("vehicle_density.tif", data_path=config.data.data_path)
    pedestrian_accessibility = optimized_load_geotiff("pedestrian_accessibility.tif", data_path=config.data.data_path)
    
    # Try to load observed data if available
    use_observed_data = config.data.use_observed_data
    observed_pedestrian_data = {}
    
    if use_observed_data:
        try:
            observed_pedestrian_morning = optimized_load_geotiff("observed_pedestrian_morning.tif", data_path=config.data.data_path)
            observed_pedestrian_afternoon = optimized_load_geotiff("observed_pedestrian_afternoon.tif", data_path=config.data.data_path) 
            observed_pedestrian_evening = optimized_load_geotiff("observed_pedestrian_evening.tif", data_path=config.data.data_path)
            
            observed_pedestrian_data = {
                "morning": observed_pedestrian_morning,
                "afternoon": observed_pedestrian_afternoon, 
                "evening": observed_pedestrian_evening
            }
            print("Observed pedestrian data loaded successfully")
        except Exception as e:
            print(f"Could not load observed data: {e}")
            use_observed_data = False
    
    # Prepare sun parameters for batch shadow calculation
    sun_params = {
        time_period: (config.times_of_day[time_period].sun_altitude, 
                     config.times_of_day[time_period].sun_azimuth)
        for time_period in time_periods
    }
    
    # Calculate shadows for all time periods at once
    print("Calculating shadow effects for all time periods...")
    shadow_intensities = batch_shadow_calculation(building_height, tree_height, sun_params)
    
    # Store results for animation
    probability_results = {}
    
    # Process for each time of day
    for time_of_day in time_periods:
        print(f"\nProcessing {time_of_day} scenario...")
        
        # Get shadow intensities for this time period
        shadow_intensity = shadow_intensities[time_of_day]
        shadow_pedestrian_attraction = calculate_shadow_pedestrian_attraction(
            shadow_intensity, 
            temperature_factor=config.times_of_day[time_of_day].temperature_factor
        )
        
        # Visualize shadow intensity
        visualize_raster(shadow_intensity, f"Shadow Intensity - {time_of_day}", 
                       f"{config.data.result_path}/shadow_intensity_{time_of_day}.png")
        
        # Learn coefficients or use defaults
        print("Determining model coefficients...")
        urbs_coeffs = config.model.urbs_coeffs
        civitas_coeffs = config.model.civitas_coeffs
        
        if use_observed_data and config.model.learn_coeffs_from_data:
            observed_data = observed_pedestrian_data[time_of_day]
            
            # Features for learning
            urbs_features = [
                building_density, (1-street_centrality), shadow_intensity, 
                poi_density, wall_constraint, (1-mean_depth), 
                isovist, vehicle_density
            ]
            urbs_features_pa = [f * pedestrian_accessibility for f in urbs_features]
            civitas_features = [pedestrian_density, shadow_pedestrian_attraction]
            
            # Learn coefficients - quantum approach if enabled
            if config.model.use_quantum_coeffs:
                try:
                    print("Learning coefficients using quantum computing...")
                    
                    # Check if API token is provided when real device is requested
                    if config.quantum.use_real_device and not config.quantum.api_token:
                        print("WARNING: You've requested to use real quantum hardware but no API token is provided.")
                        print("Falling back to simulator mode.")
                        use_real_device = False
                    else:
                        use_real_device = config.quantum.use_real_device
                        
                    # Create a scenario-specific results directory
                    quantum_results_dir = os.path.join(config.data.result_path, "quantum_results", time_of_day)
                    os.makedirs(quantum_results_dir, exist_ok=True)
                    
                    # Initialize the QuantumCoefficientOptimizer with configuration settings
                    qco = QuantumCoefficientOptimizer(
                        n_urbs_features=len(urbs_features_pa),
                        n_civitas_features=len(civitas_features),
                        use_real_device=use_real_device,  # Use the config setting
                        ibm_quantum_token=config.quantum.api_token,  # Use API token from config
                        resilience_level=config.quantum.optimization_level,  # Use optimization level from config
                        results_dir=quantum_results_dir,  # Set proper results directory
                        
                    )
                    
                    # Log which mode we're using
                    if use_real_device:
                        print("Using IBM Quantum hardware for coefficient optimization")
                    else:
                        print("Using quantum simulator for coefficient optimization")
                        
                    # Fit the model with the scenario name
                    urbs_coeffs, civitas_coeffs = qco.fit(
                        urbs_features_pa, 
                        civitas_features, 
                        observed_data, 
                        max_iterations=5,
                        scenario_name=time_of_day  # Use time of day as scenario name
                    )
                
                    # Add a delay after quantum coefficient optimization
                    if config.quantum.use_real_device:
                        print("Completed quantum coefficient optimization job.")
                        job_delay_func()

                    print(f"Quantum-optimized V_urbs coefficients: {urbs_coeffs}")
                    print(f"Quantum-optimized V_civitas coefficients: {civitas_coeffs}")
                    
                    # Additional validation against observed data
                    if use_observed_data:
                        matched_pred, matched_obs = ensure_same_shape(
                            apply_coefficients_vectorized(urbs_features_pa, urbs_coeffs) + 
                            apply_coefficients_vectorized(civitas_features, civitas_coeffs),
                            observed_data
                        )
                        print(f"Visualizing coefficient validation results in: {quantum_results_dir}")
                        
                except Exception as e:
                    print(f"Quantum coefficient optimization failed: {e}")
                    print("Falling back to classical coefficient learning...")
                    urbs_coeffs = learn_coefficients(urbs_features_pa, observed_data)
                    civitas_coeffs = learn_coefficients(civitas_features, observed_data)
                    print(f"Learned V_urbs coefficients: {urbs_coeffs}")
                    print(f"Learned V_civitas coefficients: {civitas_coeffs}")
            else:
                # Classical coefficient learning
                print("Learning coefficients from observed data...")
                urbs_coeffs = learn_coefficients(urbs_features_pa, observed_data)
                civitas_coeffs = learn_coefficients(civitas_features, observed_data)
                print(f"Learned V_urbs coefficients: {urbs_coeffs}")
                print(f"Learned V_civitas coefficients: {civitas_coeffs}")
        
        # Compute potentials with vectorized operations
        print("Computing urban potential fields...")
        
        # Prepare feature arrays
        urbs_features = [
            building_density, (1-street_centrality), shadow_intensity, 
            poi_density, wall_constraint, (1-mean_depth), 
            isovist, vehicle_density
        ]
        
        # Apply pedestrian accessibility
        urbs_features = [f * pedestrian_accessibility for f in urbs_features]
        
        # Apply coefficients in vectorized manner
        V_urbs = apply_coefficients_vectorized(urbs_features, urbs_coeffs)
        V_civitas = apply_coefficients_vectorized(
            [pedestrian_density, shadow_pedestrian_attraction], 
            civitas_coeffs
        )
        
        V_total = V_urbs + V_civitas
        V_total += 0.05

        # Visualize potential fields
        visualize_raster(V_urbs, f"Urban Potential (V_urbs) - {time_of_day}", 
                        f"{config.data.result_path}/V_urbs_{time_of_day}.png")
        visualize_raster(V_civitas, f"Social Potential (V_civitas) - {time_of_day}", 
                        f"{config.data.result_path}/V_civitas_{time_of_day}.png")
        visualize_raster(V_total, f"Total Potential Field - {time_of_day}", 
                        f"{config.data.result_path}/V_total_{time_of_day}.png")
        
        # Solve for pedestrian probability distribution
        print("Solving for pedestrian probability distribution...")
        if config.quantum.use_quantum:
            print("Using quantum approach...")
            probability = solve_schrodinger_quantum(
                V_total, 
                use_real_device=config.quantum.use_real_device,
                backend_name=None,
                job_delay_func=job_delay_func  # Pass the delay function here
            )
            
            # Add a delay after quantum job if using real device
            if config.quantum.use_real_device:
                print("Completed quantum Schrödinger equation solution job.")
                job_delay_func()
        else:
            print("Using classical Boltzmann approach...")
            probability = solve_classical_boltzmann(
                V_total, 
                temperature=config.times_of_day[time_of_day].temperature_factor
            )

        print("Applying Gaussian smoothing to enhance patterns...")
        probability = gaussian_filter(probability, sigma=0.8)
        
        # Store for animation
        probability_results[time_of_day] = probability
        
        # Enhanced visualization
        result_dict = {'data': probability}
        
        # Add comparison data if available
        if use_observed_data:
            result_dict['comparison'] = observed_pedestrian_data[time_of_day]
            
        advanced_visualization(
            result_dict,
            f"Pedestrian Probability - {time_of_day} ({'Quantum' if config.quantum.use_quantum else 'Classical'})",
            f"{config.data.result_path}/advanced_pedestrian_{time_of_day}.png"
        )
        
        # Save results as GeoTIFFs
        save_geotiff(V_urbs, "building_density.tif", f"V_urbs_{time_of_day}.tif", 
                    data_path=config.data.data_path, result_path=config.data.result_path)
        save_geotiff(V_civitas, "building_density.tif", f"V_civitas_{time_of_day}.tif", 
                    data_path=config.data.data_path, result_path=config.data.result_path)
        save_geotiff(V_total, "building_density.tif", f"V_total_{time_of_day}.tif", 
                    data_path=config.data.data_path, result_path=config.data.result_path)
        save_geotiff(shadow_intensity, "building_density.tif", f"shadow_intensity_{time_of_day}.tif", 
                    data_path=config.data.data_path, result_path=config.data.result_path)
        
        # Enhanced version - add these lines before saving
        print(f"Enhancing pedestrian patterns for better visualization...")
        enhanced_probability = enhance_patterns(probability, enhancement_factor=2.5)
        
        # Then save both the original and enhanced versions
        save_geotiff(
            probability,
            "building_density.tif",
            f"pedestrian_probability_{time_of_day}.tif",
            data_path=config.data.data_path,
            result_path=config.data.result_path
        )

        # Also save the enhanced version
        save_geotiff(
            enhanced_probability,
            "building_density.tif",
            f"pedestrian_probability_enhanced_{time_of_day}.tif",
            data_path=config.data.data_path,
            result_path=config.data.result_path
        )
        
        # Calculate correlation with observed data if available
        if use_observed_data:
            # Ensure probability and observed data have the same shape before correlation
            matched_probability, matched_observed = ensure_same_shape(
                probability, observed_pedestrian_data[time_of_day]
            )
            
            # Now calculate correlation with matched shapes
            corr, _ = pearsonr(matched_probability.flatten(), matched_observed.flatten())
            print(f"Correlation with observed data for {time_of_day}: {corr:.4f}")
            
            # Compare with classical solution
            classical_prob = solve_classical_boltzmann(V_total)
            matched_classical, matched_observed = ensure_same_shape(
                classical_prob, observed_pedestrian_data[time_of_day]
            )
            classical_corr, _ = pearsonr(matched_classical.flatten(), matched_observed.flatten())
            print(f"Classical solution correlation: {classical_corr:.4f}")
            
            if corr > classical_corr:
                print(f"Quantum approach shows {((corr/classical_corr)-1)*100:.1f}% improvement over classical")
            else:
                print(f"Classical approach performs better by {((classical_corr/corr)-1)*100:.1f}%")
        
        # Save coefficients separately to avoid dimension mismatch
        urbs_coeffs_array = np.array(urbs_coeffs)
        civitas_coeffs_array = np.array(civitas_coeffs)
        
        np.savetxt(
            os.path.join(config.data.result_path, f"urbs_coeffs_{time_of_day}.txt"),
            urbs_coeffs_array,
            header="Urban coefficients"
        )
        np.savetxt(
            os.path.join(config.data.result_path, f"civitas_coeffs_{time_of_day}.txt"),
            civitas_coeffs_array,
            header="Civitas coefficients"
        )
        
        # Also save as JSON for easier loading
        import json
        coeffs_dict = {
            "urbs_coeffs": urbs_coeffs_array.tolist(),
            "civitas_coeffs": civitas_coeffs_array.tolist()
        }
        with open(os.path.join(config.data.result_path, f"coefficients_{time_of_day}.json"), 'w') as f:
            json.dump(coeffs_dict, f, indent=4)

    # Create animation if we have multiple time periods
    if len(time_periods) >= 2:
        print("\nCreating pedestrian flow animation...")
        try:
            create_pedestrian_flow_animation(
                probability_results,
                time_periods,
                os.path.join(config.data.result_path, "pedestrian_flow_animation.gif")
            )
        except Exception as e:
            print(f"Error creating animation: {e}")
    
    print("\nEnhanced Quantum Urban Model processing complete!")
    return probability_results

# Main execution block
if __name__ == "__main__":
    # Load configuration, defaulting to "config.yaml" if no path specified
    config_path = os.environ.get("QUANTUM_URBAN_CONFIG", "config.yaml")
    try:
        config = Config.from_yaml(config_path)
        print(f"Loaded configuration from {config_path}")
    except Exception as e:
        print(f"Failed to load config from {config_path}, using defaults: {str(e)}")
        config = Config()
    
    # Run the model with the loaded configuration
    run_enhanced_quantum_urban_model(config=config)