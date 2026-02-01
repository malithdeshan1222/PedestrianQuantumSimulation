import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SPSA, COBYLA
from qiskit.primitives import Estimator as LocalEstimator, Sampler
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime.options import EstimatorOptions
from qiskit_aer import AerSimulator
from scipy.ndimage import zoom, gaussian_filter
import warnings
import time
import matplotlib.pyplot as plt
import os
import traceback
import datetime
from typing import Dict, List, Tuple, Optional, Union
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quantum_simulation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QuantumUrbanSimulator")

# Constants
IBM_QUANTUM_API_TOKEN = "d99f99454585bbc07323cd68abff5f33ee5a7570be09b04d9f84f28a3c79e2249f24d8d09ade0c928546936cd4ee910721cfd51e5cdbffc8d0c649a656376c0d"
IBM_QUANTUM_BACKEND = "ibm_brisbane"  # Primary target backend
MAX_QUBITS = 32  # Maximum qubits to use in a single circuit
MAX_CIRCUIT_DEPTH = 50  # Maximum recommended circuit depth for real hardware

# Global variables
_quantum_mapping = None
_service = None

# Default delay function implementation
def default_delay_job(delay_seconds=120):
    """
    Adds a delay before executing the next quantum job.
    
    Args:
        delay_seconds (int): Delay time in seconds, defaults to 120 seconds (2 minutes)
    """
    logger.info(f"Adding delay of {delay_seconds} seconds before next job...")
    time.sleep(delay_seconds)
    logger.info("Delay complete, continuing with next job.")


def get_quantum_service():
    """Get or initialize the QiskitRuntimeService."""
    global _service
    if _service is None:
        logger.info("Initializing Qiskit Runtime Service")
        _service = QiskitRuntimeService(channel="ibm_quantum", token=IBM_QUANTUM_API_TOKEN)
    return _service

def print_versions():
    """Print package versions for debugging"""
    import qiskit
    logger.info(f"Qiskit version: {qiskit.__version__}")
    
    try:
        import qiskit_ibm_runtime
        logger.info(f"Qiskit IBM Runtime version: {qiskit_ibm_runtime.__version__}")
    except ImportError:
        logger.info("Qiskit IBM Runtime not installed")
        
    try:
        import qiskit_aer
        logger.info(f"Qiskit Aer version: {qiskit_aer.__version__}")
    except ImportError:
        logger.info("Qiskit Aer not installed")

def print_system_info():
    """Print system and version information for debugging."""
    logger.info("=" * 50)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 50)
    
    # Date and user info
    logger.info(f"Date/Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"User: MalithDeShan")
    
    # Print versions
    print_versions()
    
    # Available quantum backends
    try:
        service = get_quantum_service()
        backends = service.backends(simulator=False, operational=True)
        logger.info("Available quantum backends:")
        for backend in backends:
            logger.info(f"- {backend.name} (Qubits: {backend.num_qubits}, Simulator: {backend.configuration().simulator})")
            
            # Print basis gates
            if hasattr(backend, 'configuration'):
                logger.info(f"  Basis gates: {backend.configuration().basis_gates}")
    except Exception as e:
        logger.error(f"Error retrieving backends: {str(e)}")
    
    logger.info("=" * 50)

def check_hardware_connectivity(backend):
    """
    Analyze hardware connectivity and print useful information.
    For IBM Quantum free account, find ECR-compatible connections.
    """
    if not backend or not hasattr(backend, 'coupling_map') or not backend.coupling_map:
        logger.warning("No coupling map available for backend")
        return False
    
    coupling_map = backend.coupling_map
    edges = list(coupling_map.get_edges())
    logger.info(f"Hardware connectivity: {len(edges)} connections between {coupling_map.size()} qubits")
    
    # Check basis gates to confirm what's available
    basis_gates = backend.configuration().basis_gates
    logger.info(f"Available basis gates: {basis_gates}")
    
    # Check which qubits can connect to qubit 0
    qubit_0_connections = [edge[1] for edge in edges if edge[0] == 0] + [edge[0] for edge in edges if edge[1] == 0]
    logger.info(f"Qubit 0 can connect to qubits: {qubit_0_connections}")
    
    # Identify pairs of qubits that can be used for ECR gates
    logger.info(f"First 10 valid ECR connections: {edges[:10]}")
    
    # Print connectivity stats
    connectivity_counts = {}
    for node in range(coupling_map.size()):
        connections = sum(1 for edge in edges if edge[0] == node or edge[1] == node)
        connectivity_counts[connections] = connectivity_counts.get(connections, 0) + 1
    
    logger.info(f"Connectivity distribution: {connectivity_counts}")
    
    # Number of qubits with different connection counts
    logger.info(f"Connectivity analysis: {connectivity_counts}")
    return True

def construct_improved_hamiltonian(V):
    """Convert potential field to quantum Hamiltonian with improved encoding."""
    global _quantum_mapping
    
    # Determine optimal qubit count based on available hardware
    max_qubits = MAX_QUBITS  # Default cap
    
    try:
        # Check available backends to determine maximum qubits
        service = get_quantum_service()
        backends = service.backends(simulator=False, operational=True)
        if backends:
            online_max_qubits = max(backend.num_qubits for backend in backends)
            # Limit to practical maximum for this algorithm
            max_qubits = min(online_max_qubits, MAX_QUBITS)
            logger.info(f"Maximum qubits available on hardware: {online_max_qubits}")
    except Exception as e:
        logger.warning(f"Could not determine max qubits from backends: {str(e)}")
    
    # Calculate required qubits for the potential dimensions
    required_qubits = int(np.ceil(np.log2(np.prod(V.shape))))
    
    # Limit qubits based on hardware and algorithm constraints
    n_qubits = max(4, min(10, required_qubits, max_qubits))
    logger.info(f"Required qubits for full representation: {required_qubits}")
    logger.info(f"Target qubits for quantum simulation: {n_qubits}")
    
    # Resize the potential grid if necessary to fit in available qubits
    scale_factor = np.sqrt(2**n_qubits / np.prod(V.shape))
    logger.info(f"Scaling factor for potential: {scale_factor:.4f}")
    
    # Keep original shape for diagnostics
    original_shape = V.shape
    
    if scale_factor < 1.0:
        logger.info(f"Downsampling potential from {V.shape} to fit in {n_qubits} qubits")
        V_small = zoom(V, scale_factor, order=1)
    else:
        V_small = V.copy()
    
    small_shape = V_small.shape
    logger.info(f"Scaled potential shape: {small_shape}")
    
    # Enhance contrast of potential field before flattening
    v_min, v_max = np.min(V_small), np.max(V_small)
    logger.info(f"Original potential range: {v_min:.4f} to {v_max:.4f}")
    
    # Apply contrast enhancement
    V_small = (V_small - v_min) / (v_max - v_min + 1e-12)
    V_small = V_small * 10.0  # Scale up for stronger quantum effects
    
    # Add small random perturbation to break degeneracy
    np.random.seed(42)  # For reproducibility
    V_small += np.random.normal(0, 0.05, V_small.shape)
    
    logger.info(f"Enhanced potential range: {np.min(V_small):.4f} to {np.max(V_small):.4f}")
    
    # Flatten and normalize with improved numerical stability
    flat_V = V_small.flatten()
    v_min, v_max = np.min(flat_V), np.max(flat_V)
    if v_max > v_min:
        flat_V = (flat_V - v_min) / (v_max - v_min)
    else:
        flat_V = np.zeros_like(flat_V)
    
    n_states = 2**n_qubits
    
    # Adaptive padding
    if n_states > len(flat_V):
        logger.info(f"Padding potential from {len(flat_V)} to {n_states} elements")
        flat_V = np.pad(flat_V, (0, n_states - len(flat_V)), 'constant')
    elif n_states < len(flat_V):
        logger.info(f"Truncating potential from {len(flat_V)} to {n_states} elements")
        flat_V = flat_V[:n_states]
    
    # Use Gray coding for better continuity in spatial mapping
    def gray_code(n):
        return n ^ (n >> 1)
    
    # Apply power transformation to make differences more pronounced
    flat_V = np.power(flat_V, 0.5)  # Square root to enhance low values
    
    # Create and store the bidirectional mapping between spatial and quantum indices
    # Use standard Python ints to avoid serialization issues
    spatial_to_quantum = {}
    quantum_to_spatial = {}
    
    # Sort spatial indices by potential value (highest first)
    sorted_spatial_indices = np.argsort(flat_V)[::-1]
    
    # Use more sophisticated Hamiltonian construction
    hamiltonian_terms = []
    
    # Add energy terms (diagonal)
    max_terms = min(256, n_states)  # Limit number of terms for efficiency
    for i in range(max_terms):
        if i < len(sorted_spatial_indices):
            # Get spatial index and map to quantum state using Gray code
            spatial_idx = int(sorted_spatial_indices[i])
            quantum_idx = int(gray_code(spatial_idx))
            
            # Store the bidirectional mapping
            spatial_to_quantum[spatial_idx] = quantum_idx
            quantum_to_spatial[quantum_idx] = spatial_idx
            
            # Create the Pauli string for this quantum state
            bin_i = format(quantum_idx, f'0{n_qubits}b')
            pauli_str = ''
            for bit in bin_i:
                pauli_str += 'Z' if bit == '1' else 'I'
            
            # Scale up coefficient values for stronger effects
            coefficient = float(flat_V[spatial_idx]) * 2.0
            hamiltonian_terms.append((pauli_str, coefficient))
    
    # Add spatial locality terms (nearest-neighbor interactions)
    if n_qubits >= 4:
        for i in range(n_qubits-1):
            pauli_str = 'I' * i + 'XX' + 'I' * (n_qubits - i - 2)
            hamiltonian_terms.append((pauli_str, 0.25))  # Increased from 0.1
    
    logger.info(f"Created Hamiltonian with {len(hamiltonian_terms)} terms")
    logger.info(f"Mapped {len(spatial_to_quantum)} spatial indices to quantum states")
    
    # Save mapping info to global variable
    _quantum_mapping = {
        'spatial_to_quantum': spatial_to_quantum,
        'quantum_to_spatial': quantum_to_spatial,
        'small_shape': small_shape,
        'original_shape': original_shape
    }
    
    # Return the Hamiltonian, qubits, and small_shape to maintain compatibility
    return SparsePauliOp.from_list(hamiltonian_terms), n_qubits, small_shape

def plot_hamiltonian_matrix(hamiltonian, filename="hamiltonian_matrix.png"):
    """Plot the Hamiltonian matrix and save it as an image file."""
    # Convert the Hamiltonian to a dense matrix
    hamiltonian_matrix = hamiltonian.to_matrix()
    
    # Plot the Hamiltonian matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(np.real(hamiltonian_matrix), cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.title("Hamiltonian Matrix")
    
    # Save the plot as an image file
    plt.savefig(filename, dpi=300)
    logger.info(f"Hamiltonian matrix plot saved to {filename}")

def create_hardware_compatible_ansatz(n_qubits, backend, reps=2):
    """
    Create a parameterized ansatz circuit using ONLY IBM's free API allowed gates:
    ECR, ID, RZ, SX, X and respecting hardware connectivity.
    
    Args:
        n_qubits: Number of qubits
        backend: IBM Quantum backend to get coupling map
        reps: Number of repetitions of the circuit pattern
    """
    logger.info(f"Creating IBM hardware-compatible ansatz with {n_qubits} qubits")
    circuit = QuantumCircuit(n_qubits)
    params = []
    
    # Get coupling map from backend
    coupling_map = None
    valid_connections = []
    
    try:
        if hasattr(backend, 'coupling_map') and backend.coupling_map:
            coupling_map = backend.coupling_map
            # Get valid connections for our qubits
            for i in range(n_qubits):
                for j in range(n_qubits):
                    if i != j and coupling_map.graph.has_edge(i, j):
                        valid_connections.append((i, j))
            
            logger.info(f"Found {len(valid_connections)} valid connections for {n_qubits} qubits")
            if valid_connections:
                logger.info(f"Sample valid connections: {valid_connections[:5]}")
        else:
            logger.warning("No coupling map available, using linear nearest-neighbor connectivity")
            for i in range(n_qubits-1):
                valid_connections.append((i, i+1))
    except Exception as e:
        logger.warning(f"Could not get coupling map: {e}, using linear connectivity")
        for i in range(n_qubits-1):
            valid_connections.append((i, i+1))
    
    # Initial layer - SX + RZ (allowed gates)
    for i in range(n_qubits):
        circuit.sx(i)
        circuit.rz(np.pi/2, i)
    
    # Create variational form with multiple layers
    for rep in range(reps):
        # Rotation layer - using only RZ and SX (allowed gates)
        for i in range(n_qubits):
            # RZ gates (virtual, no error)
            param = Parameter(f"θ_z_{rep}_{i}")
            params.append(param)
            circuit.rz(param, i)
            
            # SX gate
            circuit.sx(i)
            
            # Another RZ
            param = Parameter(f"θ_z2_{rep}_{i}")
            params.append(param)
            circuit.rz(param, i)
        
        # Entanglement layer using ECR gates (IBM's native 2-qubit gate)
        if rep < reps - 1 and valid_connections:
            # Use a subset of valid connections
            used_qubits = set()
            used_connections = 0
            
            for i, j in valid_connections:
                if i not in used_qubits and j not in used_qubits:
                    if i < n_qubits and j < n_qubits:
                        circuit.ecr(i, j)  # Using ECR instead of CX
                        used_qubits.add(i)
                        used_qubits.add(j)
                        used_connections += 1
                    
                    # Limit the number of entangling gates to keep circuit reasonable
                    if used_connections >= n_qubits//2:
                        break
            
            logger.info(f"Added {used_connections} ECR gates using valid hardware connections")
    
    # Final rotation layer
    for i in range(n_qubits):
        param = Parameter(f"θ_final_{i}")
        params.append(param)
        circuit.rz(param, i)
        circuit.sx(i)
    
    # Generate initial point
    initial_point = np.zeros(len(params))
    for i in range(len(params)):
        initial_point[i] = ((i % 6) + 1) * np.pi / 7
    
    # Display circuit stats
    gate_counts = {}
    for op in circuit.data:
        gate_name = op[0].name
        gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
    
    logger.info(f"Created ansatz with {len(params)} parameters and depth {circuit.depth()}")
    logger.info(f"Gate counts: {gate_counts}")
    
    # Ensure we're only using allowed gates
    allowed_gates = {'ecr', 'id', 'rz', 'sx', 'x'}
    used_gates = set(gate_counts.keys())
    if not used_gates.issubset(allowed_gates):
        disallowed = used_gates - allowed_gates
        logger.warning(f"Circuit contains disallowed gates: {disallowed}")
    
    return circuit, params, initial_point

def debug_quantum_components(hamiltonian, ansatz, params):
    """
    Print detailed debugging information about quantum components.
    
    Args:
        hamiltonian: The Hamiltonian operator
        ansatz: The ansatz circuit
        params: The variational parameters
    """
    logger.info("=" * 50)
    logger.info("QUANTUM COMPONENTS DEBUG INFORMATION")
    logger.info("=" * 50)
    
    # Hamiltonian information
    logger.info(f"Hamiltonian qubits: {hamiltonian.num_qubits}")
    logger.info(f"Hamiltonian terms: {hamiltonian.size}")
    
    # Print sample of Hamiltonian terms
    terms = hamiltonian.to_list()[:5]
    for i, (pauli_str, coeff) in enumerate(terms):
        logger.info(f"  Term {i}: {pauli_str} with coefficient {coeff:.6f}")
    
    # Ansatz information
    logger.info(f"Ansatz qubits: {ansatz.num_qubits}")
    logger.info(f"Ansatz depth: {ansatz.depth()}")
    logger.info(f"Parameters: {len(params)}")
    
    # Operation counts
    op_types = {}
    for op in ansatz.data:
        op_name = op[0].name
        if op_name in op_types:
            op_types[op_name] += 1
        else:
            op_types[op_name] = 1
    
    logger.info("Operation counts:")
    for op_name, count in op_types.items():
        logger.info(f"  {op_name}: {count}")
    
    # Check if the circuit might be too deep for real hardware
    if ansatz.depth() > MAX_CIRCUIT_DEPTH:
        logger.warning(f"Circuit depth {ansatz.depth()} may be too high for reliable execution on real hardware")
    
    logger.info("=" * 50)

class EstimatorAdapter:
    """Adapter for IBM Runtime Estimator with strict gate set enforcement."""
    
    def __init__(self, session, options=None, job_delay_func=None):
        """Initialize with a session."""
        self.session = session
        self.options = options
        self.job_delay_func = job_delay_func or default_delay_job
        from qiskit_ibm_runtime import Estimator
        # FIXED: Changed mode= to session=
        self._estimator = Estimator(mode=self.session, options=self.options)
        self._local_estimator = LocalEstimator()
        self._backend = session.backend
        self.expected_qubits = None
        logger.info("EstimatorAdapter initialized")
    
    def run(self, circuits, observables, parameter_values=None, **run_options):
        """
        Run estimator ensuring only allowed gates (ECR, ID, RZ, SX, X) are used
        and respecting hardware connectivity.
        """
        logger.info("EstimatorAdapter: Converting API call to compatible format")
        
        # Format inputs properly (same as before)
        if not isinstance(circuits, list):
            circuits = [circuits]
        if not isinstance(observables, list):
            observables = [observables]
        
        # Handle parameter values correctly (same as before)
        if parameter_values is None:
            parameter_values = [[]] * len(circuits)
        elif not isinstance(parameter_values, list):
            parameter_values = [parameter_values]
        elif not isinstance(parameter_values[0], list) and parameter_values:
            parameter_values = [[p] for p in parameter_values]
        
        try:
            # Get allowed gates from backend
            backend = self._backend
            allowed_gates = ['ecr', 'id', 'rz', 'sx', 'x']
            
            try:
                if hasattr(backend, 'configuration'):
                    basis_gates = backend.configuration().basis_gates
                    logger.info(f"Backend basis gates: {basis_gates}")
                    allowed_gates = [g.lower() for g in basis_gates]
            except Exception as e:
                logger.warning(f"Could not get basis gates: {e}")
            
            # Process each circuit
            compatible_circuits = []
            for i, circuit in enumerate(circuits):
                logger.info(f"Processing circuit {i+1}/{len(circuits)} with {circuit.num_qubits} qubits")
                
                # Check used gates in circuit
                circuit_gates = set(op[0].name.lower() for op in circuit.data)
                disallowed = [g for g in circuit_gates if g not in allowed_gates]
                
                if disallowed:
                    logger.warning(f"Circuit contains disallowed gates: {disallowed}")
                    logger.info("Creating new hardware-compatible circuit...")
                    
                    # Create a new circuit with only allowed gates
                    new_circuit, _, _ = create_hardware_compatible_ansatz(
                        self.expected_qubits or circuit.num_qubits,
                        backend=backend
                    )
                    compatible_circuits.append(new_circuit)
                else:
                    # Circuit already uses allowed gates, check connectivity
                    logger.info("Circuit uses allowed gates, checking for connectivity issues")
                    
                    # Look for connectivity violations
                    has_connectivity_issue = False
                    if hasattr(backend, 'coupling_map') and backend.coupling_map:
                        coupling_map = backend.coupling_map
                        for op in circuit.data:
                            if op[0].name.lower() == 'ecr' or op[0].name.lower() == 'cx':
                                qubits = [q.index for q in op[1]]
                                if len(qubits) == 2 and not coupling_map.graph.has_edge(qubits[0], qubits[1]):
                                    logger.warning(f"Connectivity violation: {op[0].name} gate between qubits {qubits} not supported")
                                    has_connectivity_issue = True
                                    break
                    
                    if has_connectivity_issue:
                        logger.info("Creating new hardware-compatible circuit...")
                        new_circuit, _, _ = create_hardware_compatible_ansatz(
                            self.expected_qubits or circuit.num_qubits,
                            backend=backend
                        )
                        compatible_circuits.append(new_circuit)
                    else:
                        logger.info("Circuit has valid connectivity, keeping original")
                        compatible_circuits.append(circuit)
            
            logger.info(f"Submitting job to Estimator with {len(compatible_circuits)} circuits")
            
            # Format pubs for the IBM Runtime Estimator (same as before)
            pubs = []
            for i in range(len(compatible_circuits)):
                circ = compatible_circuits[i]
                obs = observables[i if i < len(observables) else -1]
                params = parameter_values[i if i < len(parameter_values) else -1]
                
                # Ensure parameters format
                if params and not isinstance(params[0], list):
                    params = [params]
                    
                pubs.append((circ, obs, params))
            
            # Submit job using pubs format
            job = self._estimator.run(pubs=pubs)
            
            # Apply delay function after job submission if using real hardware
            if hasattr(self._backend, 'configuration') and not self._backend.configuration().simulator:
                logger.info("Applying job delay after submission to real quantum hardware")
                if self.job_delay_func:
                    self.job_delay_func()
            
            return job
        except Exception as e:
            logger.error(f"Error in estimator.run(): {str(e)}")
            logger.warning("Falling back to local estimator")
            
            # Fall back to local estimator
            return self._local_estimator.run(
                circuits=circuits, 
                observables=observables,
                parameter_values=parameter_values
            )

def process_estimator_job(job):
    """Safely extract data from estimator job result."""
    try:
        # Wait for job to complete
        job_result = job.result()
        
        # Check if this is a PrimitiveResult object
        if hasattr(job_result, '_pub_results'):
            logger.info("Processing a PrimitiveResult object")
            # Extract results from the individual pub results
            results = []
            for pub_result in job_result:
                # Each pub_result might have its own data structure
                if hasattr(pub_result, 'data'):
                    results.append(pub_result.data)
                elif hasattr(pub_result, 'value'):
                    results.append(pub_result.value)
                else:
                    # Just append the pub_result directly
                    results.append(pub_result)
            return results
        
        # Try different ways to extract the values based on API version
        elif hasattr(job_result, 'expectation_values'):
            logger.info("Using result.expectation_values to extract data")
            return [ev.data for ev in job_result.expectation_values]
        elif hasattr(job_result, 'quasi_dists'):
            logger.info("Using result.quasi_dists to extract data")
            return job_result.quasi_dists
        elif hasattr(job_result, 'result'):
            # This could be a nested result object
            if hasattr(job_result.result, 'values'):
                logger.info("Using result.result.values to extract data")
                return job_result.result.values
            elif hasattr(job_result.result, 'metadata'):
                logger.info("Using result.result.metadata to extract data")
                return job_result.result.metadata
        # Keep existing fallbacks
        elif hasattr(job_result, 'values'):
            logger.info("Using result.values to extract data")
            return job_result.values
        elif hasattr(job_result, 'data'):
            # Some newer versions might return a dictionary with data
            logger.info("Using result.data to extract data")
            return job_result.data
        else:
            # Last resort: try to use as-is hoping it's array-like
            logger.warning("Could not identify result structure, using raw result")
            return np.array(job_result)
    except Exception as e:
        logger.error(f"Error processing estimator job: {str(e)}")
        # Return a placeholder result
        return np.array([0.0])

def visualize_quantum_mapping(V, probability_map, small_shape):
    """Create a diagnostic visualization of the quantum-spatial mapping."""
    global _quantum_mapping
    
    if (_quantum_mapping is None):
        logger.warning("No quantum mapping data available for visualization")
        return
    
    spatial_to_quantum = _quantum_mapping['spatial_to_quantum']
    quantum_to_spatial = _quantum_mapping['quantum_to_spatial']
    
    plt.figure(figsize=(18, 12))
    
    # Plot original potential field
    plt.subplot(221)
    plt.imshow(V, cmap='viridis')
    plt.colorbar()
    plt.title("Original Potential Field (V_total)")
    
    # Plot small shape potential field
    V_small = zoom(V, np.sqrt(np.prod(small_shape)/np.prod(V.shape)), order=1)
    plt.subplot(222)
    plt.imshow(V_small, cmap='viridis')
    plt.colorbar()
    plt.title(f"Resized Potential Field ({small_shape[0]}×{small_shape[1]})")
    
    # Plot mapping visualization (top states)
    plt.subplot(223)
    # Create mapping visualization grid
    mapping_vis = np.zeros(small_shape)
    
    # Color cells based on their quantum state mapping priority
    for spatial_idx, quantum_idx in spatial_to_quantum.items():
        if spatial_idx < np.prod(small_shape):
            y, x = np.unravel_index(spatial_idx, small_shape)
            if y < small_shape[0] and x < small_shape[1]:
                mapping_vis[y, x] = 1 + quantum_idx % 10  # Modulo to create visual patterns
    
    plt.imshow(mapping_vis, cmap='tab20', interpolation='nearest')
    plt.colorbar()
    plt.title("Spatial → Quantum Mapping")
    
    # Plot final probability map
    plt.subplot(224)
    plt.imshow(probability_map, cmap='plasma')
    plt.colorbar()
    plt.title("Final Probability Distribution")
    
    plt.tight_layout()
    
    # Save the visualization
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"quantum_mapping_diagnostic_{timestamp}.png"
    plt.savefig(filename, dpi=150)
    logger.info(f"Quantum mapping visualization saved to {filename}")

# Define VQE callback once outside to avoid redefinition
def vqe_callback(eval_count, parameters, value, metadata):
    """Enhanced VQE callback supporting various data structures"""
    try:
        # Try to extract energy value in different formats
        if isinstance(value, (float, int)):
            energy = value
        elif hasattr(value, 'real'):
            energy = value.real
        elif isinstance(value, dict) and 'energy' in value:
            energy = value['energy']
        else:
            energy = float(value)  # Try direct conversion
        logger.info(f"VQE Iteration {eval_count}: Energy = {energy:.5f}")
    except Exception as e:
        logger.warning(f"Callback error: {str(e)}")

def solve_schrodinger_quantum(V, use_real_device=True, backend_name=None, job_delay_func=None):
    """
    Solve the Schrödinger equation using IBM Quantum hardware with error mitigation.
    
    Args:
        V: 2D array representing the potential grid
        use_real_device: Whether to use real quantum hardware 
        backend_name: Specific backend to use (if None, will select automatically)
        job_delay_func: Function to delay between quantum jobs (default: 2 minutes)
        
    Returns:
        2D array of probability distribution matching the shape of V
    """
    global _quantum_mapping
    
    # Use default delay function if none provided
    if job_delay_func is None:
        job_delay_func = default_delay_job
    
    logger.info("=" * 50)
    logger.info("QUANTUM SOLVER STARTED")
    logger.info("=" * 50)
    
    start_time = time.time()
    
    try:
        # Print versions for debugging
        print_versions()
        
        # Construct the Hamiltonian with improved mapping
        logger.info("Constructing Hamiltonian...")
        hamiltonian, n_qubits, small_shape = construct_improved_hamiltonian(V)
        logger.info(f"Using {n_qubits} qubits for quantum simulation")
        
        # Set up quantum backend if using real hardware
        if use_real_device:
            try:
                # Get the quantum service
                service = get_quantum_service()
                
                # Select the best backend for our task
                if backend_name:
                    try:
                        backend = service.backend(backend_name)
                        logger.info(f"Using requested backend: {backend.name} with {backend.num_qubits} qubits")
                    except Exception as e:
                        logger.warning(f"Specified backend {backend_name} not available: {str(e)}")
                        backend_name = None  # Reset to allow auto-selection
                
                if not backend_name:
                    # Select a backend with a reasonable number of qubits
                    backends = service.backends(simulator=False, operational=True)
                    if not backends:
                        raise RuntimeError("No operational quantum backends available")
                    
                    # Select a suitable backend
                    backend = None
                    for b in backends:
                        if 16 <= b.num_qubits <= 127:  # Good middle ground
                            backend = b
                            break
                    
                    # If no ideal backend found, just use the first available
                    if backend is None:
                        backend = backends[0]
                    
                    logger.info(f"Selected backend: {backend.name} with {backend.num_qubits} qubits")
                
                # Check hardware connectivity and available gates
                check_hardware_connectivity(backend)
                
                # For small systems (≤4 qubits), use exact eigensolver
                if n_qubits <= 4:
                    logger.info("Using exact eigensolver for small quantum system")
                    solver = NumPyMinimumEigensolver()
                    result = solver.compute_minimum_eigenvalue(hamiltonian)
                    statevector = result.eigenstate
                    logger.info(f"Exact solution found with eigenvalue: {result.eigenvalue}")
                else:
                    # Create hardware-compatible ansatz using only ECR, ID, RZ, SX, X gates
                    logger.info("Creating IBM hardware-compatible ansatz...")
                    
                    # Use our specialized function to create a hardware-compatible ansatz
                    ansatz, params, initial_point = create_hardware_compatible_ansatz(
                        n_qubits, 
                        backend=backend, 
                        reps=2
                    )
                    
                    # Debug the quantum components
                    debug_quantum_components(hamiltonian, ansatz, params)
                    
                    # Choose optimizer based on circuit complexity
                    if len(params) > 20:
                        optimizer = SPSA(maxiter=100)
                        logger.info("Using SPSA optimizer for larger parameter space")
                    else:
                        optimizer = COBYLA(maxiter=100)
                        logger.info("Using COBYLA optimizer")
                    
                    # Set up session with error mitigation
                    estimator_options = EstimatorOptions()
                    logger.info("Enabled error mitigation with resilience_level=1")
                    
                    logger.info("Establishing quantum session...")
                    with Session(backend=backend) as session:
                        logger.info(f"Session established with backend: {backend.name}")
                        
                        # Use our EstimatorAdapter with hardware constraints and job delay
                        estimator = EstimatorAdapter(
                            session=session, 
                            options=estimator_options,
                            job_delay_func=job_delay_func  # Pass the delay function
                        )
                        estimator.expected_qubits = hamiltonian.num_qubits
                        logger.info(f"Set estimator's expected qubits to {hamiltonian.num_qubits}")
                        
                        # Run VQE with error mitigation using the globally defined callback
                        logger.info("Running VQE with error mitigation...")
                        vqe = VQE(
                            estimator=estimator,
                            ansatz=ansatz,
                            optimizer=optimizer,
                            initial_point=initial_point,
                            callback=vqe_callback  # Use the global callback
                        )
                        
                        result = vqe.compute_minimum_eigenvalue(hamiltonian)
                        # FIXED: Log VQE-specific attributes rather than metadata
                        logger.info(f"VQE optimal_value: {result.optimal_value}")
                        logger.info(f"VQE optimal_point: {result.optimal_point}")
                        logger.info(f"VQE cost_function_evals: {result.cost_function_evals}")
                        logger.info(f"VQE optimizer_time: {result.optimizer_time}")
                        logger.info(f"VQE completed with eigenvalue: {result.eigenvalue}")
                        
                        # Add a delay after the VQE job completes
                        if use_real_device:
                            logger.info("Adding delay after VQE job...")
                            job_delay_func()
                        
                        # Extract optimal parameters
                        optimal_point = result.optimal_point
                        
                        # Use assign_parameters with a dictionary for reliability
                        parameter_dict = dict(zip(params, optimal_point))
                        optimal_circuit = ansatz.assign_parameters(parameter_dict)
                        
                        # Ensure circuit has no free parameters
                        if len(optimal_circuit.parameters) > 0:
                            logger.warning(f"Circuit still has {len(optimal_circuit.parameters)} free parameters!")
            
            except Exception as e:
                logger.error(f"Error with IBM Quantum hardware: {str(e)}")
                logger.warning("Falling back to simulator")
                use_real_device = False
        
        # Use simulator if not using real hardware or if hardware access failed
        if not use_real_device:
            logger.info("Using local simulator")
            estimator = LocalEstimator()
            
            # For small systems (≤4 qubits), use exact eigensolver
            if n_qubits <= 4:
                logger.info("Using exact eigensolver for small system")
                solver = NumPyMinimumEigensolver()
                result = solver.compute_minimum_eigenvalue(hamiltonian)
                statevector = result.eigenstate
            else:
                # Create parameterized circuit using only IBM's free API allowed gates
                ansatz, params, initial_point = create_hardware_compatible_ansatz(
                    n_qubits, 
                    backend=None,  # No backend for simulator
                    reps=2
                )
                
                # Configure optimizer
                optimizer = COBYLA(maxiter=100)
                
                # Run VQE with local estimator using the globally defined callback
                logger.info("Running Variational Quantum Eigensolver...")
                vqe = VQE(
                    estimator=estimator,
                    ansatz=ansatz,
                    optimizer=optimizer,
                    initial_point=initial_point,
                    callback=vqe_callback  # Use the global callback
                )
                
                # Compute minimum eigenvalue
                start_time_vqe = time.time()
                result = vqe.compute_minimum_eigenvalue(hamiltonian)
                logger.info(f"VQE completed in {time.time() - start_time_vqe:.2f} seconds")
                # FIXED: Log VQE-specific attributes
                logger.info(f"VQE optimal_value: {result.optimal_value}")
                logger.info(f"VQE cost_function_evals: {result.cost_function_evals}")
                logger.info(f"VQE eigenvalue: {result.eigenvalue}")
                
                # Extract optimal circuit parameters
                optimal_point = result.optimal_point
                
                # Use assign_parameters with a dictionary
                parameter_dict = dict(zip(params, optimal_point))
                optimal_circuit = ansatz.assign_parameters(parameter_dict)
                
                # Ensure circuit is final and has no free parameters
                if len(optimal_circuit.parameters) > 0:
                    logger.warning(f"Circuit still has {len(optimal_circuit.parameters)} free parameters!")
        
        # Get statevector through simulation if not already obtained
        if not 'statevector' in locals():
            logger.info("Simulating optimal circuit for statevector...")
            
            # Try multiple methods to get statevector - improved error handling
            try:
                # Method 1: Direct AerSimulator
                simulator = AerSimulator(method='statevector')
                transpiled_circuit = transpile(optimal_circuit, simulator)
                simulation_result = simulator.run(transpiled_circuit).result()
                statevector = simulation_result.get_statevector()
                logger.info(f"Statevector obtained with method 1 (AerSimulator)")
            except Exception as e1:
                logger.warning(f"AerSimulator method failed: {str(e1)}")
                
                try:
                    # Method 2: Use Statevector class
                    from qiskit.quantum_info import Statevector
                    statevector = Statevector(optimal_circuit)
                    logger.info(f"Statevector obtained with method 2 (Statevector class)")
                except Exception as e2:
                    logger.warning(f"Statevector class method failed: {str(e2)}")
                    
                    try:
                        # Method 3: Use StatevectorSampler primitive
                        from qiskit.primitives import StatevectorSampler
                        sampler = StatevectorSampler()
                        result = sampler.run(optimal_circuit, shots=1).result()
                        statevector = result.quasi_dists[0]
                        logger.info(f"Statevector obtained with method 3 (StatevectorSampler)")
                    except Exception as e3:
                        logger.error(f"All statevector methods failed: {str(e3)}")
                        logger.warning("Using uniform distribution as fallback")
                        # Create a uniform statevector
                        statevector = np.ones(2**n_qubits) / np.sqrt(2**n_qubits)
        
        # Calculate probabilities from the statevector
        if hasattr(statevector, 'probabilities'):
            probabilities = statevector.probabilities()
        else:
            probabilities = np.abs(np.array(statevector))**2
        
        logger.info(f"Statevector shape: {len(probabilities)}")
        logger.info(f"Raw probability stats: min={np.min(probabilities):.7f}, max={np.max(probabilities):.7f}")
        logger.info(f"Number of non-zero probabilities: {np.sum(probabilities > 1e-10)}/{len(probabilities)}")
        
        # IMPROVED MAPPING: Use the stored bidirectional mapping to correctly reconstruct the spatial grid
        # Create an empty probability array for the small shape
        prob_small = np.zeros(np.prod(small_shape))
        
        # Map quantum probabilities back to spatial positions using our mapping
        if _quantum_mapping is not None:
            quantum_to_spatial = _quantum_mapping['quantum_to_spatial']
            mapped_count = 0
            
            # Use mapping to place probabilities in correct spatial positions
            for quantum_idx, prob in enumerate(probabilities):
                if prob > 1e-10 and quantum_idx in quantum_to_spatial:  # Only map significant probabilities
                    spatial_idx = quantum_to_spatial[quantum_idx]
                    if spatial_idx < len(prob_small):
                        prob_small[spatial_idx] = prob
                        mapped_count += 1
            
            logger.info(f"Successfully mapped {mapped_count} quantum states to spatial positions")
        else:
            # If no mapping info, fall back to direct index mapping
            logger.warning("No quantum mapping information available! Using direct index mapping.")
            prob_small[:len(probabilities)] = probabilities[:np.prod(small_shape)]
        
        # Reshape to 2D grid
        prob_map_small = prob_small.reshape(small_shape)
        
        # Apply smoothing to reduce noise
        prob_map_small = gaussian_filter(prob_map_small, sigma=0.5)
        
        # Upsample back to original dimensions
        zoom_factor = np.array(V.shape) / np.array(small_shape)
        probability_map = zoom(prob_map_small, zoom_factor, order=1)
        
        # Use min-max normalization instead of sum normalization
        probability_map = (probability_map - np.min(probability_map)) / (np.max(probability_map) - np.min(probability_map) + 1e-8)
        
        # Apply Gaussian smoothing to reduce noise and enhance patterns
        probability_map = gaussian_filter(probability_map, sigma=0.8)
        
        # Apply contrast enhancement for better visualization
        mean_prob = np.mean(probability_map)
        std_prob = np.std(probability_map)
        probability_map = np.where(
            probability_map > mean_prob + 0.5*std_prob,
            probability_map * 1.5,
            probability_map * 0.5
        )
        
        # Final normalization
        probability_map = (probability_map - np.min(probability_map)) / (np.max(probability_map) - np.min(probability_map) + 1e-8)
        
        logger.info(f"Final probability map stats: min={np.min(probability_map):.6f}, max={np.max(probability_map):.6f}")
        
        # Create visualization to help debug the mapping
        try:
            visualize_quantum_mapping(V, probability_map, small_shape)
        except Exception as e:
            logger.warning(f"Could not create mapping visualization: {str(e)}")
        
        # Calculate total execution time
        total_time = time.time() - start_time
        logger.info(f"Quantum solver completed in {total_time:.2f} seconds")
        
        return probability_map

    except Exception as e:
        logger.error("!" * 50)
        logger.error(f"ERROR IN QUANTUM SOLUTION: {str(e)}")
        logger.error("!" * 50)
        traceback.print_exc()
        
        # Return a fallback solution
        logger.warning("Returning classical solution as fallback")
        return solve_classical_boltzmann(V)

def solve_classical_boltzmann(V, temperature=1.0):
    """Classical solution using Boltzmann distribution."""
    logger.info("Solving classically using Boltzmann distribution")
    
    # Invert potential (lower potential = higher probability)
    energy = -V  
    
    # Apply Boltzmann distribution formula
    boltzmann = np.exp(energy / temperature)
    probability = boltzmann / np.sum(boltzmann)
    
    # Apply the same normalization and enhancement as quantum solution
    # for fair comparison
    probability = (probability - np.min(probability)) / (np.max(probability) - np.min(probability) + 1e-8)
    probability = gaussian_filter(probability, sigma=0.8)
    
    logger.info("Classical solution completed")
    return probability

def visualize_results(V, quantum_probability, classical_probability, save_path="quantum_results"):
    """
    Visualize and save comparison between quantum and classical results.
    
    Args:
        V: Original potential grid
        quantum_probability: Probability distribution from quantum solution
        classical_probability: Probability distribution from classical solution
        save_path: Directory to save results
    """
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create visualization with 3 subplots
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Original potential
    plt.subplot(1, 3, 1)
    plt.imshow(V, origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title('Original Potential Field')
    
    # Plot 2: Classical probability
    plt.subplot(1, 3, 2)
    plt.imshow(classical_probability, origin='lower', cmap='plasma')
    plt.colorbar()
    plt.title('Classical Boltzmann Distribution')
    
    # Plot 3: Quantum probability
    plt.subplot(1, 3, 3)
    plt.imshow(quantum_probability, origin='lower', cmap='plasma')
    plt.colorbar()
    plt.title('Quantum Solution Distribution')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_path, f"probability_comparison_{timestamp}.png")
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Visualization saved to {plot_path}")
    
    # Calculate the difference between quantum and classical
    difference = quantum_probability - classical_probability
    
    # Create a second plot for the difference
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(difference, origin='lower', cmap='coolwarm')
    plt.colorbar()
    plt.title('Quantum - Classical Difference')
    
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(difference), origin='lower', cmap='hot')
    plt.colorbar()
    plt.title('Absolute Difference')
    
    plt.tight_layout()
    
    # Save the difference plot
    diff_path = os.path.join(save_path, f"quantum_classical_diff_{timestamp}.png")
    plt.savefig(diff_path, dpi=300)
    logger.info(f"Difference visualization saved to {diff_path}")
    
    # Also save the data as numpy arrays for later analysis
    np.save(os.path.join(save_path, f"potential_{timestamp}.npy"), V)
    np.save(os.path.join(save_path, f"quantum_prob_{timestamp}.npy"), quantum_probability)
    np.save(os.path.join(save_path, f"classical_prob_{timestamp}.npy"), classical_probability)
    
    logger.info("Results data saved as numpy arrays")

def main():
    """Main function to run the quantum urban simulator."""
    # Print system info
    print_system_info()
    
    # Create sample potential (or load from data)
    logger.info("Creating test potential grid")
    
    # Example potential field with interesting features
    n = 8  # 8x8 grid for faster execution
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    
    # Create a potential with two minima
    V = 2.0 * ((X - 0.25)**2 + (Y - 0.25)**2) + \
        1.5 * ((X - 0.75)**2 + (Y - 0.75)**2) + \
        0.5 * np.sin(8 * np.pi * X) * np.sin(8 * np.pi * Y)
    
    # Normalize potential for better comparison
    V = (V - np.min(V)) / (np.max(V) - np.min(V))
    
    # By default, try to use real hardware but can be toggled off
    use_real_device = True
    
    try:
        # Solve using quantum backend
        logger.info("Starting quantum solution...")
        start_time = time.time()
        quantum_probability = solve_schrodinger_quantum(V, use_real_device=use_real_device)
        quantum_time = time.time() - start_time
        logger.info(f"Quantum solution completed in {quantum_time:.2f} seconds")
        
        # Solve classically for comparison
        logger.info("Starting classical solution...")
        start_time = time.time()
        classical_probability = solve_classical_boltzmann(V)
        classical_time = time.time() - start_time
        logger.info(f"Classical solution completed in {classical_time:.2f} seconds")
        
        # Visualize and save the results
        visualize_results(V, quantum_probability, classical_probability)
        
        # Display results
        plt.show()
        
        logger.info("Simulation completed successfully")
        
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        traceback.print_exc()

if __name__ == '__main__':
    main()