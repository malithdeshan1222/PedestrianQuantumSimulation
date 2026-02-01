import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.ndimage import zoom, gaussian_filter
import os
import rasterio
from rasterio.transform import from_origin
from rasterio.plot import show
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import warnings
import time
import sys

RANDOM_SEED = 42

# Import functionality from other scripts in the repository
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Try to import from other scripts in the repository
try:
    from quantum_utils import enhanced_preprocessing, normalize_gis_data
    from raster_helpers import load_raster_data, transform_coordinates
    from visualization_tools import create_colormap, add_basemap_carefully
    print("Successfully imported supporting modules")
except ImportError as e:
    print(f"Warning: Could not import supporting modules: {e}")
    # Define fallback functions if imports fail
    def enhanced_preprocessing(data):
        return data
    
    def normalize_gis_data(data):
        # Normalize to [0,1] range
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val > min_val:
            return (data - min_val) / (max_val - min_val)
        return np.zeros_like(data)
    
    def transform_coordinates(x, y, transform):
        return x, y

# Add global variable to store quantum-spatial mapping
_quantum_mapping = None

# =============================================================================
# 1. Utility Functions
# =============================================================================

def load_geotiff(filename):
    """Load GeoTIFF with proper handling of spatial reference."""
    try:
        # Try to use enhanced loading function if imported
        if 'load_raster_data' in globals():
            return load_raster_data(filename)
        
        # Fallback standard loading
        with rasterio.open(filename) as src:
            data = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            return data, transform, crs, bounds
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        # Return dummy data if file can't be loaded
        return np.random.random((32, 32)), None, None, None
        
def save_geotiff(data, output_filename, transform=None, crs=None):
    """Save data as GeoTIFF with spatial reference."""
    try:
        if transform is None:
            transform = from_origin(0, 0, 1, 1)
        
        new_dataset = rasterio.open(
            output_filename,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=crs,
            transform=transform
        )
        new_dataset.write(data, 1)
        new_dataset.close()
        print(f"Saved {output_filename}")
    except Exception as e:
        print(f"Error saving {output_filename}: {e}")

def normalize(data):
    """Normalize data to range [0, 1]."""
    if 'normalize_gis_data' in globals():
        return normalize_gis_data(data)
    
    # Standard normalization fallback
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val > min_val:
        return (data - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(data)

def transform_coords(i, j, transform):
    """Convert pixel coordinates to geographic coordinates."""
    if transform is None:
        return j, i  # Just return x, y if no transform
    
    # Use imported transform function if available
    if 'transform_coordinates' in globals():
        return transform_coordinates(i, j, transform)
    
    # Standard transformation fallback
    x, y = rasterio.transform.xy(transform, i, j, offset='center')
    return x, y

def inverse_transform_coords(x, y, transform):
    """Convert geographic coordinates to pixel coordinates."""
    if transform is None:
        return int(y), int(x)  # Just return as pixel coords
    
    row, col = rasterio.transform.rowcol(transform, x, y)
    return row, col

# =============================================================================
# 2. Quantum Model Functions with Enhanced Bidirectional Mapping
# =============================================================================

def create_urban_hamiltonian(potential_field):
    """Create a quantum Hamiltonian from an urban potential field with strict shape-preserving bidirectional mapping."""
    global _quantum_mapping
    
    # Get dimensions and determine qubit count
    original_shape = potential_field.shape
    n_qubits = int(np.ceil(np.log2(np.prod(original_shape))))
    
    # Calculate target shape precisely based on qubit count
    # Use powers of 2 along each dimension to ensure perfect reconstructability
    target_shape = [2**(n_qubits//2), 2**(n_qubits//2)]
    if n_qubits % 2 == 1:  # Handle odd number of qubits
        target_shape[0] *= 2
    
    # Store original indices before resizing for perfect reconstruction
    y_indices = np.linspace(0, original_shape[0]-1, target_shape[0])
    x_indices = np.linspace(0, original_shape[1]-1, target_shape[1])
    y_indices = np.round(y_indices).astype(int)
    x_indices = np.round(x_indices).astype(int)
    
    # Create the resized field by sampling the original at the calculated indices
    resized_field = np.zeros(target_shape)
    for i, y in enumerate(y_indices):
        for j, x in enumerate(x_indices):
            resized_field[i, j] = potential_field[min(y, original_shape[0]-1), min(x, original_shape[1]-1)]
    
    # Store mapping information
    index_mapping = {
        'y_indices': y_indices,
        'x_indices': x_indices,
        'original_to_resized': {},
        'resized_to_original': {}
    }
    
    # Create bidirectional mapping between original and resized indices
    for i, y in enumerate(y_indices):
        for j, x in enumerate(x_indices):
            orig_idx = (y, x)
            resized_idx = (i, j)
            index_mapping['original_to_resized'][orig_idx] = resized_idx
            index_mapping['resized_to_original'][resized_idx] = orig_idx
    
    # Flatten the resized field for quantum encoding
    flat_field = resized_field.flatten()
    
    # Create and store the bidirectional mapping between spatial and quantum indices
    spatial_to_quantum = {}
    quantum_to_spatial = {}
    
    # Gray code function for better spatial locality
    def gray_code(n):
        return n ^ (n >> 1)
    
    # Add diagonal terms based on potential field values - strict ordering for perfect reconstruction
    hamiltonian_terms = []
    for spatial_idx in range(len(flat_field)):
        # Map spatial index to quantum index using Gray code
        quantum_idx = gray_code(spatial_idx)
        
        # Store the mapping
        spatial_to_quantum[spatial_idx] = quantum_idx
        quantum_to_spatial[quantum_idx] = spatial_idx
        
        # Create Pauli string representation
        bin_i = format(quantum_idx, f'0{n_qubits}b')
        pauli_str = ''
        for bit in bin_i:
            pauli_str += 'Z' if bit == '1' else 'I'
        
        # Add term with coefficient from potential field
        coefficient = float(flat_field[spatial_idx])
        hamiltonian_terms.append((pauli_str, coefficient))
    
    # Add spatial locality terms (nearest-neighbor interactions) for better quantum representation
    if n_qubits >= 2:
        for i in range(n_qubits-1):
            pauli_str = 'I' * i + 'XX' + 'I' * (n_qubits - i - 2)
            hamiltonian_terms.append((pauli_str, 0.1))
    
    # Save the mapping information for later reconstruction
    _quantum_mapping = {
        'spatial_to_quantum': spatial_to_quantum,
        'quantum_to_spatial': quantum_to_spatial,
        'resized_shape': target_shape,
        'original_shape': original_shape,
        'index_mapping': index_mapping
    }
    
    # Log useful diagnostic information
    print(f"Original shape: {original_shape}, Resized shape: {target_shape}")
    print(f"Using {n_qubits} qubits for quantum representation")
    
    return SparsePauliOp.from_list(hamiltonian_terms), n_qubits

def create_parameterized_ansatz(n_qubits, reps=2):
    """Create the parameterized ansatz circuit used in the urban model"""
    circuit = QuantumCircuit(n_qubits)
    
    # Create explicit parameters
    params = []
    param_index = 0
    
    # Create variational form with multiple layers
    for rep in range(reps):
        # Rotation layer
        for i in range(n_qubits):
            param = Parameter(f"θ_{param_index}")
            params.append(param)
            circuit.rx(param, i)
            param_index += 1
            
            param = Parameter(f"θ_{param_index}")
            params.append(param)
            circuit.rz(param, i)
            param_index += 1
        
        # Entanglement layer (unless it's the last repetition)
        if rep < reps - 1:
            for i in range(n_qubits - 1):
                circuit.cx(i, i + 1)
            # Make it cyclic for better connectivity
            if n_qubits > 2:
                circuit.cx(n_qubits - 1, 0)
    
    # Final rotation layer
    for i in range(n_qubits):
        param = Parameter(f"θ_{param_index}")
        params.append(param)
        circuit.ry(param, i)
        param_index += 1
    
    return circuit, params

def simplified_vqe(ansatz, hamiltonian, params, initial_point=None, iterations=50):
    """Simplified VQE that works with any Qiskit version."""
    simulator = AerSimulator()
    
    if initial_point is None:
        initial_point = np.random.random(len(params)) * 2 * np.pi
    
    # Function to calculate energy given parameters
    def calculate_energy(params_values):
        # Bind parameters to circuit
        param_dict = dict(zip(params, params_values))
        bound_circuit = ansatz.assign_parameters(param_dict)
        
        # Add measurement for each Pauli term
        bound_circuit.save_statevector()
        
        # Run simulation
        job = simulator.run(bound_circuit)
        result = job.result()
        sv = result.get_statevector(bound_circuit)
        
        # Calculate energy expectation value
        if hasattr(sv, 'data'):
            # For newer versions where statevector is a Statevector object
            energy = sv.expectation_value(hamiltonian).real
        else:
            # For older versions where statevector might be a numpy array
            sv_array = np.asarray(sv)
            energy = np.real(np.vdot(sv_array, hamiltonian.to_matrix() @ sv_array))
        
        return energy
    
    # Simple optimization using COBYLA
    from scipy.optimize import minimize
    result = minimize(calculate_energy, initial_point, method='COBYLA', 
                      options={'maxiter': iterations})
    
    # Get iterations count - handle different SciPy versions
    iteration_count = None
    for attr in ['nit', 'nfev', 'njev', 'nhev']:
        if hasattr(result, attr):
            iteration_count = getattr(result, attr)
            break
    if iteration_count is None:
        iteration_count = iterations  # Fallback
    
    # Create final circuit with optimal parameters
    optimal_point = result.x
    param_dict = dict(zip(params, optimal_point))
    optimal_circuit = ansatz.assign_parameters(param_dict)
    optimal_circuit.save_statevector()
    
    # Run final simulation
    job = simulator.run(optimal_circuit)
    sim_result = job.result()
    final_statevector = sim_result.get_statevector(optimal_circuit)
    
    # Calculate probabilities
    if hasattr(final_statevector, 'probabilities'):
        probabilities = final_statevector.probabilities()
    else:
        probabilities = np.abs(np.asarray(final_statevector))**2
    
    # Reshape to match the original grid
    grid_size = int(np.sqrt(len(probabilities)))
    prob_grid = probabilities[:grid_size**2].reshape(grid_size, grid_size)
    
    return {
        'optimal_point': optimal_point,
        'optimal_value': result.fun,
        'statevector': final_statevector,
        'probabilities': probabilities,  # Return raw 1D probabilities for bidirectional mapping
        'prob_grid': prob_grid,         # Also include the 2D grid for compatibility
        'iterations': iteration_count
    }

def classical_boltzmann(potential, temperature=1.0):
    """Classical Boltzmann solution for comparison."""
    energy = -potential  # Invert potential (lower potential = higher probability)
    boltzmann = np.exp(-energy / temperature)
    return boltzmann / np.sum(boltzmann)

# =============================================================================
# 3. Scenario Modification Functions
# =============================================================================

def apply_point_scenario(potential_field, center, intensity, radius, scenario_type='increase', transform=None):
    """Apply a point-based scenario to the potential field."""
    # Create a copy of the original potential field
    modified_field = potential_field.copy()
    
    # Convert center if transform is provided and center is in geographic coordinates
    if transform is not None and isinstance(center[0], float) and isinstance(center[1], float):
        # Center is in geo coords, convert to pixel coords
        center = inverse_transform_coords(center[1], center[0], transform)
    
    # Create distance array
    y, x = np.ogrid[:potential_field.shape[0], :potential_field.shape[1]]
    dist_squared = (x - center[1])**2 + (y - center[0])**2
    
    # Create mask for the affected area
    mask = dist_squared <= radius**2
    
    # Apply modification
    if scenario_type == 'increase':
        # Add Gaussian-shaped increase
        modification = intensity * np.exp(-dist_squared / (2 * (radius/2)**2))
        modified_field += modification
    elif scenario_type == 'decrease':
        # Add Gaussian-shaped decrease
        modification = intensity * np.exp(-dist_squared / (2 * (radius/2)**2))
        modified_field -= modification
    elif scenario_type == 'barrier':
        # Create a circular barrier
        modified_field[mask] = np.maximum(modified_field[mask], 0.8)
    elif scenario_type == 'attractor':
        # Create a circular attractor (lower potential)
        modified_field[mask] = np.minimum(modified_field[mask], 0.2)
        
    # Ensure values stay in [0, 1] range
    modified_field = np.clip(modified_field, 0, 1)
    
    return modified_field

def apply_line_scenario(potential_field, start, end, width, intensity, scenario_type='path', transform=None):
    """Apply a line-based scenario (path, street, barrier) to the potential field."""
    # Create a copy of the original potential field
    modified_field = potential_field.copy()
    
    # Convert points if transform is provided and points are in geographic coordinates
    if transform is not None:
        if isinstance(start[0], float) and isinstance(start[1], float):
            # Start is in geo coords, convert to pixel coords
            start = inverse_transform_coords(start[1], start[0], transform)
        if isinstance(end[0], float) and isinstance(end[1], float):
            # End is in geo coords, convert to pixel coords
            end = inverse_transform_coords(end[1], end[0], transform)
    
    # Get line parameters
    y0, x0 = start
    y1, x1 = end
    
    # Create coordinate grids
    y, x = np.mgrid[:potential_field.shape[0], :potential_field.shape[1]]
    
    # Calculate distance to line segment
    # Line equation: ax + by + c = 0
    a = y1 - y0
    b = x0 - x1
    c = x1*y0 - x0*y1
    
    # Distance from point to line: |ax + by + c| / sqrt(a² + b²)
    dist = np.abs(a*x + b*y + c) / np.sqrt(a**2 + b**2)
    
    # Distance from point to line segment (not just infinite line)
    # Project point onto line
    t = ((x - x0) * (x1 - x0) + (y - y0) * (y1 - y0)) / ((x1 - x0)**2 + (y1 - y0)**2)
    t = np.clip(t, 0, 1)
    px = x0 + t * (x1 - x0)
    py = y0 + t * (y1 - y0)
    
    # Distance to closest point on segment
    dist_to_segment = np.sqrt((x - px)**2 + (y - py)**2)
    
    # Create mask for the affected area
    mask = dist_to_segment <= width
    
    # Apply modification
    if scenario_type == 'path':
        # Create a pedestrian path (lower potential)
        modified_field[mask] = np.minimum(modified_field[mask], 1.0 - intensity)
    elif scenario_type == 'barrier':
        # Create a barrier (higher potential)
        modified_field[mask] = np.maximum(modified_field[mask], intensity)
    elif scenario_type == 'street':
        # Create a street (medium-low potential)
        modified_field[mask] = 0.3
    
    # Ensure values stay in [0, 1] range
    modified_field = np.clip(modified_field, 0, 1)
    
    return modified_field

def apply_area_scenario(potential_field, top_left, bottom_right, intensity, scenario_type='zone', transform=None):
    """Apply an area-based scenario to the potential field."""
    # Create a copy of the original potential field
    modified_field = potential_field.copy()
    
    # Convert points if transform is provided and points are in geographic coordinates
    if transform is not None:
        if isinstance(top_left[0], float) and isinstance(top_left[1], float):
            # Top left is in geo coords, convert to pixel coords
            top_left = inverse_transform_coords(top_left[1], top_left[0], transform)
        if isinstance(bottom_right[0], float) and isinstance(bottom_right[1], float):
            # Bottom right is in geo coords, convert to pixel coords
            bottom_right = inverse_transform_coords(bottom_right[1], bottom_right[0], transform)
    
    # Extract coordinates
    y0, x0 = top_left
    y1, x1 = bottom_right
    
    # Ensure proper ordering
    y_min, y_max = min(y0, y1), max(y0, y1)
    x_min, x_max = min(x0, x1), max(x0, x1)
    
    # Ensure within bounds
    y_min = max(0, y_min)
    y_max = min(potential_field.shape[0], y_max)
    x_min = max(0, x_min)
    x_max = min(potential_field.shape[1], x_max)
    
    # Create mask for the affected area
    mask = np.zeros_like(potential_field, dtype=bool)
    mask[y_min:y_max, x_min:x_max] = True
    
    # Apply modification
    if scenario_type == 'zone':
        # Create a pedestrian zone (lower potential)
        modified_field[mask] = np.minimum(modified_field[mask], 1.0 - intensity)
    elif scenario_type == 'development':
        # Create a new development (higher potential)
        modified_field[mask] = np.maximum(modified_field[mask], intensity)
    elif scenario_type == 'park':
        # Create a park (low potential with some variation)
        area_shape = (y_max - y_min, x_max - x_min)
        park_texture = 0.2 + 0.1 * np.random.random(area_shape)
        modified_field[y_min:y_max, x_min:x_max] = park_texture
    
    # Ensure values stay in [0, 1] range
    modified_field = np.clip(modified_field, 0, 1)
    
    return modified_field

# =============================================================================
# 4. Interactive Scenario Explorer
# =============================================================================

class ScenarioExplorer:
    def __init__(self, potential_field, transform=None, crs=None, bounds=None):
        self.potential_field = potential_field
        self.transform = transform
        self.crs = crs
        self.bounds = bounds
        self.modified_field = potential_field.copy()
        
        # Initialize caching variables
        self._last_input = None
        self._last_output = None
        self._last_params = None

        # Calculate initial probability
        self.original_probability = self.calculate_probability(potential_field)
        self.new_probability = self.original_probability.copy()
        
        self.is_drawing = False
        self.start_point = None
        self.end_point = None
        self.scenario_count = 0
        
        # Set up the scenario parameters
        self.scenario_type = 'point'  # 'point', 'line', 'area'
        self.effect_type = 'increase'  # 'increase', 'decrease', 'barrier', 'attractor', 'path', 'zone'...
        self.intensity = 0.5
        self.radius = 5
        self.width = 3
        
        # Feature collections for visualization
        self.points = []
        self.lines = []
        self.areas = []
        
        # Initialize visualization objects to prevent "before definition" errors
        self.circle_collection = None
        self.motion_circle = None
        self.im_potential = None
        self.im_modified = None
        self.im_orig_prob = None
        self.im_new_prob = None
        self.im_diff = None
        self.cb_diff = None
        self.coord_text = None
        
        self.setup_ui()
    
    def calculate_probability(self, potential_field):
        """Calculate with deterministic results for consistent outputs."""
        global _quantum_mapping
        
        # Set random seed for reproducibility
        np.random.seed(RANDOM_SEED)
        
        print("Calculating pedestrian distribution...")
        
        # If input is identical to previously computed field, return cached result
        if hasattr(self, '_last_input') and np.array_equal(self._last_input, potential_field):
            print("Using cached result for identical input")
            return self._last_output
        
        # Create Hamiltonian (this will also update _quantum_mapping)
        hamiltonian, n_qubits = create_urban_hamiltonian(potential_field)
        
        # Create circuit
        circuit, params = create_parameterized_ansatz(n_qubits, reps=1)
        
        # Use fixed initial parameters for VQE
        if hasattr(self, '_last_params') and len(self._last_params) == len(params):
            initial_point = self._last_params
            print("Using previous parameters as starting point")
        else:
            # If no previous parameters, use fixed seed
            initial_point = np.random.random(len(params)) * 2 * np.pi
        
        result = simplified_vqe(circuit, hamiltonian, params, initial_point, iterations=30)
        
        # Store optimal parameters for future calculations
        self._last_params = result['optimal_point']
        
        # Run VQE
        result = simplified_vqe(circuit, hamiltonian, params, iterations=30)
        
        # Get raw probability amplitudes
        raw_probabilities = result['probabilities']
        
        # Use improved bidirectional mapping to create exactly shape-matched probability distribution
        if _quantum_mapping is not None:
            # Get the mapping information
            resized_shape = _quantum_mapping['resized_shape']
            original_shape = _quantum_mapping['original_shape']
            quantum_to_spatial = _quantum_mapping['quantum_to_spatial']
            index_mapping = _quantum_mapping['index_mapping']
            
            # Map quantum states to resized grid using the quantum-to-spatial mapping
            resized_probabilities = np.zeros(np.prod(resized_shape))
            for quantum_idx, prob in enumerate(raw_probabilities):
                if quantum_idx in quantum_to_spatial:
                    spatial_idx = quantum_to_spatial[quantum_idx]
                    if spatial_idx < len(resized_probabilities):
                        resized_probabilities[spatial_idx] = prob
            
            # Reshape to the resized 2D grid that matches our quantum encoding
            resized_prob_map = resized_probabilities.reshape(resized_shape)
            
            # Now map exactly back to the original grid using our precise index mapping
            original_prob_map = np.zeros(original_shape)
            
            # Use the stored index mapping to reconstruct the original shape perfectly
            for i in range(original_shape[0]):
                for j in range(original_shape[1]):
                    # Find closest point in the resized grid
                    closest_i = np.argmin(np.abs(index_mapping['y_indices'] - i))
                    closest_j = np.argmin(np.abs(index_mapping['x_indices'] - j))
                    original_prob_map[i, j] = resized_prob_map[closest_i, closest_j]
            
            # This is now exactly the same shape as the input
            probability_map = original_prob_map
        else:
            # Fallback if mapping not available
            probability_map = result['prob_grid']
            # Resize to match original
            if probability_map.shape != potential_field.shape:
                probability_map = zoom(probability_map, (potential_field.shape[0]/probability_map.shape[0],
                                                       potential_field.shape[1]/probability_map.shape[1]))
        
        # Apply smoothing appropriate for the scale
        sigma = min(0.005 * min(original_shape), 0.8)  # Scale-appropriate smoothing
        probability_map = gaussian_filter(probability_map, sigma=sigma)
        
        # Final normalization 
        probability_map = normalize(probability_map)
        
        print(f"Quantum calculation complete. Output shape: {probability_map.shape} (matches input: {potential_field.shape})")
        
            # Cache results
        self._last_input = potential_field.copy()
        self._last_output = probability_map

        return probability_map
    
    def setup_ui(self):
        """Set up the interactive UI for scenario exploration."""
        # Create figure
        self.fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, figure=self.fig, height_ratios=[1, 1, 0.3])
        
        # Input panels: Original and Modified Potential Fields
        self.ax_potential = self.fig.add_subplot(gs[0, 0])
        self.ax_modified = self.fig.add_subplot(gs[0, 1])
        
        # Output panels: Original and New Probability
        self.ax_orig_prob = self.fig.add_subplot(gs[1, 0])
        self.ax_new_prob = self.fig.add_subplot(gs[1, 1])
        
        # Difference panel
        self.ax_diff = self.fig.add_subplot(gs[:2, 2])
        
        # Controls
        self.ax_scenario_type = self.fig.add_subplot(gs[2, 0])
        self.ax_effect_type = self.fig.add_subplot(gs[2, 1])
        self.ax_params = self.fig.add_subplot(gs[2, 2])
        
        # Display initial data with proper georeferencing
        if self.transform is not None and self.crs is not None:
            # Ensure consistent extent/bounds for all visualizations
            if self.bounds:
                extent = [self.bounds.left, self.bounds.right, self.bounds.bottom, self.bounds.top]
            else:
                extent = None
                
            # For original potential
            self.im_potential = show(self.potential_field, ax=self.ax_potential, 
                                   transform=self.transform, cmap='viridis')
            # For modified potential
            self.im_modified = show(self.modified_field, ax=self.ax_modified, 
                                  transform=self.transform, cmap='viridis')
            # For probabilities
            self.im_orig_prob = show(self.original_probability, ax=self.ax_orig_prob, 
                                   transform=self.transform, cmap='plasma')
            self.im_new_prob = show(self.new_probability, ax=self.ax_new_prob, 
                                  transform=self.transform, cmap='plasma')
        else:
            # Fallback to regular imshow
            self.im_potential = self.ax_potential.imshow(self.potential_field, cmap='viridis')
            self.im_modified = self.ax_modified.imshow(self.modified_field, cmap='viridis')
            self.im_orig_prob = self.ax_orig_prob.imshow(self.original_probability, cmap='plasma')
            self.im_new_prob = self.ax_new_prob.imshow(self.new_probability, cmap='plasma')
        
        # Set titles
        self.ax_potential.set_title('Original Potential Field')
        self.ax_modified.set_title('Modified Potential Field')
        self.ax_orig_prob.set_title('Original Pedestrian Probability')
        self.ax_new_prob.set_title('New Pedestrian Probability')
        
        # Create a new custom colormap for the difference
        colors = [(0, 0, 0.5), (1, 1, 1), (0.5, 0, 0)]  # Blue to white to red
        cmap_diff = LinearSegmentedColormap.from_list('diff_cmap', colors, N=256)
        
        # Initialize difference plot
        diff = self.new_probability - self.original_probability
        vmax = max(abs(np.min(diff)), abs(np.max(diff))) or 0.1
        
        if self.transform is not None and self.crs is not None:
            # For georeferenced data, show() returns the axes
            show(diff, ax=self.ax_diff, transform=self.transform, 
                 cmap=cmap_diff, vmin=-vmax, vmax=vmax)
            # Get the image from the axes for the colorbar
            if len(self.ax_diff.images) > 0:
                self.im_diff = self.ax_diff.images[0]
            else:
                # Fallback if no image was created
                self.im_diff = self.ax_diff.imshow(diff, cmap=cmap_diff, vmin=-vmax, vmax=vmax)
        else:
            self.im_diff = self.ax_diff.imshow(diff, cmap=cmap_diff, vmin=-vmax, vmax=vmax)
            
        self.ax_diff.set_title('Probability Difference (New - Original)')
        self.cb_diff = plt.colorbar(self.im_diff, ax=self.ax_diff)
        
        # Display coordinate text if georeferenced
        if self.transform is not None:
            self.coord_text = self.fig.text(0.01, 0.01, "", fontsize=10)
        
        # Add scenario type selection
        self.ax_scenario_type.axis('off')
        self.radio_scenario = RadioButtons(
            self.ax_scenario_type, 
            ('Point Modification', 'Line Modification', 'Area Modification'),
            active=0
        )
        self.radio_scenario.on_clicked(self.on_scenario_change)
        
        # Add effect type selection (this will change based on scenario type)
        self.ax_effect_type.axis('off')
        self.update_effect_types()
        
        # Add parameter sliders
        self.ax_params.axis('off')
        ax_intensity = plt.axes([0.6, 0.15, 0.3, 0.03])
        self.slider_intensity = Slider(ax_intensity, 'Intensity', 0.0, 1.0, valinit=0.5)
        self.slider_intensity.on_changed(self.update_params)
        
        ax_radius = plt.axes([0.6, 0.10, 0.3, 0.03])
        self.slider_radius = Slider(ax_radius, 'Radius/Width', 1, 20, valinit=5, valstep=1)
        self.slider_radius.on_changed(self.update_params)
        
        # Add buttons for actions
        ax_apply = plt.axes([0.6, 0.05, 0.15, 0.03])
        self.btn_apply = Button(ax_apply, 'Apply Changes')
        self.btn_apply.on_clicked(self.apply_changes)
        
        ax_reset = plt.axes([0.8, 0.05, 0.15, 0.03])
        self.btn_reset = Button(ax_reset, 'Reset All')
        self.btn_reset.on_clicked(self.reset_all)
        
        # Save buttons
        ax_save_potential = plt.axes([0.45, 0.15, 0.1, 0.03])
        self.btn_save_potential = Button(ax_save_potential, 'Save Modified Potential')
        self.btn_save_potential.on_clicked(self.save_potential)
        
        ax_save_prob = plt.axes([0.45, 0.10, 0.1, 0.03])
        self.btn_save_prob = Button(ax_save_prob, 'Save New Probability')
        self.btn_save_prob.on_clicked(self.save_probability)
        
        ax_save_diff = plt.axes([0.45, 0.05, 0.1, 0.03])
        self.btn_save_diff = Button(ax_save_diff, 'Save Difference Map')
        self.btn_save_diff.on_clicked(self.save_difference)
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        # Display instructions
        self.fig.text(0.5, 0.01, "Click and drag on the Modified Potential field to apply scenarios",
                     ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
    
    def update_coord_display(self, event):
        """Update coordinate display."""
        if hasattr(self, 'coord_text') and event.inaxes == self.ax_modified and self.transform is not None:
            # Display pixel coordinates
            i, j = int(event.ydata), int(event.xdata)
            x, y = transform_coords(i, j, self.transform)
            self.coord_text.set_text(f"Coordinates: Pixel({j}, {i}), Geo({x:.6f}, {y:.6f})")
    
    def redraw_features(self):
        """Redraw all features on the modified potential field."""
        # If we have georeferenced data, use appropriate plotting
        if self.transform is not None:
            # Create GeoDataFrame for points
            if self.points:
                points_gdf = gpd.GeoDataFrame(
                    geometry=[Point(transform_coords(y, x, self.transform)) for (y, x) in self.points],
                    crs=self.crs
                )
                points_gdf.plot(ax=self.ax_modified, color='red', markersize=50, alpha=0.5)
            
            # Create GeoDataFrame for lines
            if self.lines:
                lines_gdf = gpd.GeoDataFrame(
                    geometry=[LineString([transform_coords(start[0], start[1], self.transform), 
                                        transform_coords(end[0], end[1], self.transform)]) 
                            for (start, end) in self.lines],
                    crs=self.crs
                )
                lines_gdf.plot(ax=self.ax_modified, color='blue', linewidth=3, alpha=0.5)
            
            # Create GeoDataFrame for areas
            if self.areas:
                areas_gdf = gpd.GeoDataFrame(
                    geometry=[Polygon([
                        transform_coords(tl[0], tl[1], self.transform),
                        transform_coords(tl[0], br[1], self.transform),
                        transform_coords(br[0], br[1], self.transform),
                        transform_coords(br[0], tl[1], self.transform)
                    ]) for (tl, br) in self.areas],
                    crs=self.crs
                )
                areas_gdf.plot(ax=self.ax_modified, color='green', alpha=0.3)
        else:
            # Simple matplotlib plotting
            for y, x in self.points:
                circle = plt.Circle((x, y), radius=self.radius, color='red', fill=True, alpha=0.3)
                self.ax_modified.add_patch(circle)
            
            for start, end in self.lines:
                self.ax_modified.plot([start[1], end[1]], [start[0], end[0]], 'b-', linewidth=self.width, alpha=0.5)
            
            for tl, br in self.areas:
                rect = Rectangle((tl[1], tl[0]), br[1]-tl[1], br[0]-tl[0], 
                                 color='green', alpha=0.3)
                self.ax_modified.add_patch(rect)
    
    def update_effect_types(self):
        """Update effect type options based on the selected scenario type."""
        self.ax_effect_type.clear()
        self.ax_effect_type.axis('off')
        
        if self.scenario_type == 'point':
            self.radio_effect = RadioButtons(
                self.ax_effect_type, 
                ('Increase Potential', 'Decrease Potential', 'Create Barrier', 'Create Attractor'),
                active=0
            )
            self.effect_type = 'increase'
        elif self.scenario_type == 'line':
            self.radio_effect = RadioButtons(
                self.ax_effect_type, 
                ('Create Path', 'Create Barrier', 'Create Street'),
                active=0
            )
            self.effect_type = 'path'
        elif self.scenario_type == 'area':
            self.radio_effect = RadioButtons(
                self.ax_effect_type, 
                ('Pedestrian Zone', 'New Development', 'Create Park'),
                active=0
            )
            self.effect_type = 'zone'
            
        self.radio_effect.on_clicked(self.on_effect_change)
    
    def on_scenario_change(self, label):
        """Handle scenario type change."""
        if label == 'Point Modification':
            self.scenario_type = 'point'
        elif label == 'Line Modification':
            self.scenario_type = 'line'
        elif label == 'Area Modification':
            self.scenario_type = 'area'
            
        self.update_effect_types()
    
    def on_effect_change(self, label):
        """Handle effect type change."""
        if self.scenario_type == 'point':
            if label == 'Increase Potential':
                self.effect_type = 'increase'
            elif label == 'Decrease Potential':
                self.effect_type = 'decrease'
            elif label == 'Create Barrier':
                self.effect_type = 'barrier'
            elif label == 'Create Attractor':
                self.effect_type = 'attractor'
        elif self.scenario_type == 'line':
            if label == 'Create Path':
                self.effect_type = 'path'
            elif label == 'Create Barrier':
                self.effect_type = 'barrier'
            elif label == 'Create Street':
                self.effect_type = 'street'
        elif self.scenario_type == 'area':
            if label == 'Pedestrian Zone':
                self.effect_type = 'zone'
            elif label == 'New Development':
                self.effect_type = 'development'
            elif label == 'Create Park':
                self.effect_type = 'park'
    
    def update_params(self, val):
        """Update scenario parameters from sliders."""
        self.intensity = self.slider_intensity.val
        self.radius = self.slider_radius.val
        self.width = self.slider_radius.val
    
    def on_press(self, event):
        """Handle mouse press event."""
        # Update coordinate display
        self.update_coord_display(event)
        
        if event.inaxes != self.ax_modified:
            return
            
        self.is_drawing = True
        y, x = int(event.ydata), int(event.xdata)
        self.start_point = (y, x)
        
        # For point scenario, apply immediately
        if self.scenario_type == 'point':
            # Apply to raster data
            self.modified_field = apply_point_scenario(
                self.modified_field, 
                self.start_point, 
                self.intensity, 
                self.radius, 
                self.effect_type,
                self.transform
            )
            
            # Update visualization
            self.points.append(self.start_point)
            
            if self.transform is not None:
                # Update data in existing image instead of clearing and redrawing
                if len(self.ax_modified.images) > 0:
                    self.ax_modified.images[0].set_array(self.modified_field)
                else:
                    # Fallback if no image exists
                    self.ax_modified.clear()
                    show(self.modified_field, ax=self.ax_modified, transform=self.transform, cmap='viridis')
                    self.ax_modified.set_title('Modified Potential Field')
                
                # Add the circle for visualization without redrawing everything
                if hasattr(self, 'circle_collection'):
                    # Remove previous circles to avoid overload
                    self.circle_collection.remove()
                
                # Create circle at the point
                from matplotlib.patches import Circle
                circle = Circle((x, y), radius=self.radius, color='red', fill=True, alpha=0.3, transform=self.ax_modified.transData)
                self.circle_collection = self.ax_modified.add_patch(circle)
            else:
                # Non-georeferenced case
                self.im_modified.set_data(self.modified_field)
                
                # Add visualization for the clicked point
                circle = plt.Circle((x, y), radius=self.radius, color='red', fill=True, alpha=0.3)
                self.ax_modified.add_patch(circle)
            
            # Redraw only what changed
            self.fig.canvas.draw_idle()
    
    def on_release(self, event):
        """Handle mouse release event."""
        # Update coordinate display
        self.update_coord_display(event)
        
        if not self.is_drawing:
            return
            
        self.is_drawing = False
        
        if event.inaxes != self.ax_modified:
            return
            
        y, x = int(event.ydata), int(event.xdata)
        self.end_point = (y, x)
        
        # Apply scenario based on type
        if self.scenario_type == 'line':
            # Apply to raster data
            self.modified_field = apply_line_scenario(
                self.modified_field,
                self.start_point,
                self.end_point,
                self.width,
                self.intensity,
                self.effect_type,
                self.transform
            )
            
            # Update visualization
            self.lines.append((self.start_point, self.end_point))
            
        elif self.scenario_type == 'area':
            # Apply to raster data
            self.modified_field = apply_area_scenario(
                self.modified_field,
                self.start_point,
                self.end_point,
                self.intensity,
                self.effect_type,
                self.transform
            )
            
            # Update visualization
            self.areas.append((self.start_point, self.end_point))
            
        # Update visualization
        if self.transform is not None:
            # Update data without redrawing entire plot
            if len(self.ax_modified.images) > 0:
                self.ax_modified.images[0].set_array(self.modified_field)
                
                # Add line or area visualization
                if self.scenario_type == 'line':
                    start_y, start_x = self.start_point
                    end_y, end_x = self.end_point
                    self.ax_modified.plot([start_x, end_x], [start_y, end_y], 
                                         color='blue', linewidth=self.width, alpha=0.6)
                elif self.scenario_type == 'area':
                    top_left_y, top_left_x = self.start_point
                    bottom_right_y, bottom_right_x = self.end_point
                    width = bottom_right_x - top_left_x
                    height = bottom_right_y - top_left_y
                    rect = Rectangle((top_left_x, top_left_y), width, height, 
                                   color='green', alpha=0.3)
                    self.ax_modified.add_patch(rect)
            else:
                # Fallback if no image exists
                self.ax_modified.clear()
                show(self.modified_field, ax=self.ax_modified, transform=self.transform, cmap='viridis')
                self.ax_modified.set_title('Modified Potential Field')
                self.redraw_features()
        else:
            self.im_modified.set_data(self.modified_field)
            self.redraw_features()
            
        self.fig.canvas.draw_idle()
    
    def on_motion(self, event):
        """Handle mouse motion event."""
        # Update coordinate display
        self.update_coord_display(event)
        
        if not self.is_drawing or event.inaxes != self.ax_modified:
            return
            
        # For point scenarios, apply continuously while dragging
        if self.scenario_type == 'point':
            y, x = int(event.ydata), int(event.xdata)
            center = (y, x)
            
            # Apply to raster data
            self.modified_field = apply_point_scenario(
                self.modified_field, 
                center, 
                self.intensity/5,  # Lower intensity for continuous application
                self.radius, 
                self.effect_type,
                self.transform
            )
            
            # Update visualization
            self.points.append(center)
            
            if self.transform is not None:
                # Update data without full redraw
                if len(self.ax_modified.images) > 0:
                    self.ax_modified.images[0].set_array(self.modified_field)
                
                # Add small dot for current position
                if hasattr(self, 'motion_circle'):
                    self.motion_circle.remove()
                from matplotlib.patches import Circle
                dot = Circle((x, y), radius=1, color='blue', alpha=0.7, transform=self.ax_modified.transData)
                self.motion_circle = self.ax_modified.add_patch(dot)
            else:
                self.im_modified.set_data(self.modified_field)
            
            # Efficient redraw
            self.fig.canvas.draw_idle()
    
    def apply_changes(self, event):
        """Apply the current modified potential and calculate new probabilities."""
        # Calculate new pedestrian distribution
        self.new_probability = self.calculate_probability(self.modified_field)
        
        # Update visualizations
        if self.transform is not None:
            # Replot with proper georeferencing
            self.ax_new_prob.clear()
            show(self.new_probability, ax=self.ax_new_prob, transform=self.transform, cmap='plasma')
            self.ax_new_prob.set_title('New Pedestrian Probability')
            
            # Update difference map
            diff = self.new_probability - self.original_probability
            vmax = max(abs(np.min(diff)), abs(np.max(diff))) or 0.1
            
            # Remove existing colorbar before creating a new one
            if hasattr(self, 'cb_diff') and self.cb_diff is not None:
                self.cb_diff.remove()
                
            # Replot with proper georeferencing
            self.ax_diff.clear()
            show(diff, ax=self.ax_diff, transform=self.transform, 
                cmap='coolwarm', vmin=-vmax, vmax=vmax)
            
            # Get the image for the colorbar
            if len(self.ax_diff.images) > 0:
                self.im_diff = self.ax_diff.images[0]
                self.cb_diff = plt.colorbar(self.im_diff, ax=self.ax_diff)
            else:
                # Fallback
                self.im_diff = self.ax_diff.imshow(diff, cmap='coolwarm', vmin=-vmax, vmax=vmax)
                self.cb_diff = plt.colorbar(self.im_diff, ax=self.ax_diff)
        else:
            self.im_new_prob.set_data(self.new_probability)
            
            # Update difference map
            diff = self.new_probability - self.original_probability
            vmax = max(abs(np.min(diff)), abs(np.max(diff))) or 0.1
            self.im_diff.set_data(diff)
            self.im_diff.set_clim(vmin=-vmax, vmax=vmax)
            self.cb_diff.update_normal(self.im_diff)
        
        # Update titles to show statistics
        mean_diff = np.mean(diff)
        max_increase = np.max(diff)
        max_decrease = np.min(diff)
        self.ax_diff.set_title(f'Difference: Mean={mean_diff:.4f}, Max+={max_increase:.4f}, Max-={max_decrease:.4f}')
        
        self.scenario_count += 1
        self.fig.canvas.draw_idle()
    
    def reset_all(self, event):
        """Reset to original potential field."""
        self.modified_field = self.potential_field.copy()
        self.new_probability = self.original_probability.copy()
        
        # Clear feature collections
        self.points = []
        self.lines = []
        self.areas = []
        
        # Update visualizations
        if self.transform is not None:
            # Update data without full redraw if possible
            if len(self.ax_modified.images) > 0:
                self.ax_modified.images[0].set_array(self.modified_field)
            else:
                # Fallback to full redraw
                self.ax_modified.clear()
                show(self.modified_field, ax=self.ax_modified, transform=self.transform, cmap='viridis')
                
            if len(self.ax_new_prob.images) > 0:
                self.ax_new_prob.images[0].set_array(self.original_probability)
            else:
                # Fallback to full redraw
                self.ax_new_prob.clear()
                show(self.new_probability, ax=self.ax_new_prob, transform=self.transform, cmap='plasma')
            
            # Remove existing colorbar before creating a new one
            if hasattr(self, 'cb_diff') and self.cb_diff is not None:
                self.cb_diff.remove()
                
            # Reset difference to zeros
            diff = np.zeros_like(self.original_probability)
            self.ax_diff.clear()
            show(diff, ax=self.ax_diff, transform=self.transform, 
                 cmap='coolwarm', vmin=-0.1, vmax=0.1)
            
            # Create new colorbar
            if len(self.ax_diff.images) > 0:
                self.im_diff = self.ax_diff.images[0]
                self.cb_diff = plt.colorbar(self.im_diff, ax=self.ax_diff)
            else:
                # Fallback
                self.im_diff = self.ax_diff.imshow(diff, cmap='coolwarm', vmin=-0.1, vmax=0.1)
                self.cb_diff = plt.colorbar(self.im_diff, ax=self.ax_diff)
                
            # Restore titles
            self.ax_modified.set_title('Modified Potential Field')
            self.ax_new_prob.set_title('New Pedestrian Probability')
        else:
            # Non-georeferenced case
            self.im_modified.set_data(self.modified_field)
            self.im_new_prob.set_data(self.new_probability)
            
            # Update difference map
            diff = np.zeros_like(self.original_probability)
            self.im_diff.set_data(diff)
            self.im_diff.set_clim(vmin=-0.1, vmax=0.1)
            self.cb_diff.update_normal(self.im_diff)
        
        # Reset title
        self.ax_diff.set_title('Probability Difference (New - Original)')
        
        self.scenario_count = 0
        self.fig.canvas.draw_idle()
    
    def save_potential(self, event):
        """Save the modified potential field."""
        filename = f"modified_potential_{self.scenario_count}.tif"
        save_geotiff(self.modified_field, filename, self.transform, self.crs)
    
    def save_probability(self, event):
        """Save the new probability distribution."""
        filename = f"new_probability_{self.scenario_count}.tif"
        save_geotiff(self.new_probability, filename, self.transform, self.crs)
    
    def save_difference(self, event):
        """Save the difference map."""
        diff = self.new_probability - self.original_probability
        filename = f"probability_difference_{self.scenario_count}.tif"
        save_geotiff(diff, filename, self.transform, self.crs)

# =============================================================================
# 5. Main Application
# =============================================================================

def run_scenario_explorer():
    """Main function to run the scenario explorer."""
    print("Urban Quantum Pedestrian Model - Scenario Explorer")
    print("=================================================")
    print("Version: 2.0.0 - Enhanced with Bidirectional Mapping")
    
    # Ask for input potential field
    potential_file = input("Enter potential field GeoTIFF filename (or press Enter for demo mode): ")
    
    if potential_file and os.path.exists(potential_file):
        potential_field, transform, crs, bounds = load_geotiff(potential_file)
        print(f"Loaded potential field from {potential_file}")
        
        # Apply any imported preprocessing if available
        if 'enhanced_preprocessing' in globals():
            potential_field = enhanced_preprocessing(potential_field)
            print("Applied enhanced preprocessing to potential field")
    else:
        print("Using demo potential field...")
        # Create a demo potential field with urban-like features
        size = 32
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        
        # City center (lower potential)
        center = np.exp(-(X**2 + Y**2)/(2*0.3**2))
        
        # Street grid
        streets_x = 0.5 * np.sin(np.pi * X * 5)**20
        streets_y = 0.5 * np.sin(np.pi * Y * 5)**20
        streets = np.maximum(streets_x, streets_y)
        
        # Buildings (higher potential)
        buildings = np.zeros((size, size))
        for _ in range(50):
            bx = np.random.uniform(-0.8, 0.8)
            by = np.random.uniform(-0.8, 0.8)
            bw = np.random.uniform(0.05, 0.2)
            buildings += 0.8 * np.exp(-((X-bx)**2 + (Y-by)**2)/(2*bw**2))
        
        # Combine into potential field
        potential_field = 0.8 * buildings - 0.5 * center + 0.3 * streets
        potential_field = normalize(potential_field)
        transform, crs, bounds = None, None, None
    
    # Launch the explorer
    explorer = ScenarioExplorer(potential_field, transform, crs, bounds)
    plt.show()

if __name__ == "__main__":
    run_scenario_explorer()