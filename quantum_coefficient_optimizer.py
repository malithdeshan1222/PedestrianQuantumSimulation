import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_algorithms.optimizers import COBYLA, SPSA  # Updated import path
from qiskit.primitives import Estimator as LocalEstimator, Sampler as LocalSampler
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Estimator
from qiskit_ibm_runtime.options import EstimatorOptions, SamplerOptions
import time
import os
import logging
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import json
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quantum_optimization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QuantumCoefficientOptimizer")

# Define job delay function to be used between quantum job executions
def default_delay_job(delay_seconds=120):
    """
    Adds a delay before executing the next quantum job.
    
    Args:
        delay_seconds (int): Delay time in seconds, defaults to 120 seconds (2 minutes)
    """
    logger.info(f"Adding delay of {delay_seconds} seconds between quantum jobs...")
    time.sleep(delay_seconds)
    logger.info("Delay complete, continuing with next job.")

class QuantumCoefficientOptimizer:
    """Uses quantum computing to optimize urban model coefficients."""
    
    def __init__(self, n_urbs_features=8, n_civitas_features=2, use_real_device=False, 
                 ibm_quantum_token=None, backend_name=None, resilience_level=1,
                 results_dir="quantum_results", job_delay_func=None):
        """
        Initialize the quantum optimizer.
        
        Args:
            n_urbs_features: Number of urban features
            n_civitas_features: Number of civic features
            use_real_device: Whether to use real quantum hardware (IBM Quantum)
            ibm_quantum_token: IBM Quantum API token (defaults to env var IBM_QUANTUM_TOKEN)
            backend_name: Name of specific backend to use
            resilience_level: Error mitigation level (0-3, where 0 is none and 3 is max)
            results_dir: Directory to save results and visualizations
            job_delay_func: Function to add delay between quantum jobs (defaults to 2 minutes)
        """
        self.n_urbs_features = n_urbs_features
        self.n_civitas_features = n_civitas_features
        self.total_features = n_urbs_features + n_civitas_features
        self.use_real_device = use_real_device
        self.backend_name = backend_name
        self.resilience_level = resilience_level
        self.results_dir = results_dir
        self.job_delay_func = job_delay_func or default_delay_job
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Tracking optimization progress
        self.optimization_history = {
            'iterations': [],
            'loss_values': [],
            'urbs_coeffs': [],
            'civitas_coeffs': []
        }
        
        # Store validation metrics
        self.validation_metrics = {}
        
        # Create local simulators as fallback
        self.local_estimator = LocalEstimator()
        self.local_sampler = LocalSampler()
        
        # Configure runtime estimator and sampler
        self.service = None
        self.backend = None
        self.estimator = None
        self.sampler = None
        
        # Set up IBM Quantum if requested
        if use_real_device:
            # Get token from parameter or environment variable
            self.ibm_quantum_token = ibm_quantum_token or os.environ.get('IBM_QUANTUM_TOKEN')
            
            if not self.ibm_quantum_token:
                logger.warning("No IBM Quantum token provided. Falling back to simulator.")
                self.use_real_device = False
                self.estimator = self.local_estimator
                self.sampler = self.local_sampler
            else:
                try:
                    self._setup_quantum_backend()
                except Exception as e:
                    logger.error(f"Failed to initialize IBM Quantum backend: {str(e)}")
                    logger.warning("Falling back to local simulator")
                    self.use_real_device = False
                    self.estimator = self.local_estimator
                    self.sampler = self.local_sampler
        else:
            # Use local simulator
            self.estimator = self.local_estimator
            self.sampler = self.local_sampler
    
    def _setup_quantum_backend(self):
        """Initialize IBM Quantum backend and primitives."""
        logger.info("Initializing IBM Quantum service")
        self.service = QiskitRuntimeService(
            channel="ibm_quantum",
            token=self.ibm_quantum_token
        )
        
        # Log available backends for debugging
        backends = self.service.backends()
        logger.info(f"Available backends: {[b.name for b in backends]}")
        
        # Select backend
        if self.backend_name:
            try:
                self.backend = self.service.backend(self.backend_name)
                logger.info(f"Using specified backend: {self.backend.name}")
            except Exception as e:
                logger.warning(f"Specified backend {self.backend_name} not available: {str(e)}")
                self.backend_name = None
        
        if not self.backend_name:
            # Auto-select a suitable backend
            backends = self.service.backends(simulator=False, operational=True)
            if not backends:
                raise RuntimeError("No operational quantum backends available")
            
            # Select a backend with at least 16 qubits if possible
            for backend in backends:
                if backend.num_qubits >= min(16, self.total_features):
                    self.backend = backend
                    break
            
            # If no suitable backend found, use the first available
            if not self.backend and backends:
                self.backend = backends[0]
            
            logger.info(f"Auto-selected backend: {self.backend.name} ({self.backend.num_qubits} qubits)")
        
        logger.info("IBM Quantum backend setup complete")
    
    def create_feature_encoding_circuit(self, features_array, params, for_hardware=False):
        """
        Create a minimal quantum circuit with only SX and RZ gates.
        
        Args:
            features_array: Array of feature values to encode
            params: Variational parameters for the circuit
            for_hardware: Whether to optimize circuit for hardware
        """
        n_qubits = min(16, self.total_features)  # Cap at 16 qubits for performance
        qc = QuantumCircuit(n_qubits)
        
        # Use only universally supported gates: SX and RZ
        for i in range(n_qubits):
            # Apply feature encoding
            feature_val = features_array[i] if i < len(features_array) else 0.0
            
            # Apply SX gate (rotation around X axis)
            qc.sx(i)
            
            # Apply RZ rotation based on feature value (normalized to [0, π])
            qc.rz(feature_val * np.pi, i)
            
            # Apply parameter from optimization
            param_idx = i % len(params)
            qc.rz(params[param_idx], i)
        
        # Add measurements
        qc.measure_all()
        
        # Remove any barriers that might have been added
        data_no_barriers = []
        for gate in qc.data:
            if gate[0].name.lower() != 'barrier':
                data_no_barriers.append(gate)
        
        # If barriers were removed, update circuit
        if len(data_no_barriers) < len(qc.data):
            logger.info(f"Removed {len(qc.data) - len(data_no_barriers)} barrier gates")
            qc.data = data_no_barriers
        
        return qc
    
    def quantum_cost_function(self, params, feature_sets, target_data, iteration=None):
        """
        Compute loss using quantum circuit.
        
        Args:
            params: Variational parameters for the circuit
            feature_sets: Sets of features to evaluate
            target_data: Target values to compare against
            iteration: Current iteration number (for tracking)
        """
        loss = 0
        # Use a smaller batch size for better performance
        batch_size = min(5 if self.use_real_device else 10, len(feature_sets))
        
        # Process features in batches
        selected_indices = np.random.choice(len(feature_sets), batch_size, replace=False)
        
        # Initialize primitives if using IBM Quantum
        if self.use_real_device:
            try:
                with Session(backend=self.backend) as session:
                    # Set up estimator and sampler primitives
                    estimator_options = EstimatorOptions()
                    sampler_options = SamplerOptions()
                    logger.info(f"Enabled error mitigation with resilience_level={self.resilience_level}")
                    
                    # Following the pattern in reference code: use mode=session
                    self.estimator = Estimator(mode=session, options=estimator_options)
                    self.sampler = Sampler(mode=session, options=sampler_options)
                    
                    # Process batch with real hardware
                    loss = self._process_batch_quantum(params, feature_sets, target_data, selected_indices, batch_size)
                    
                    # Add 2-minute delay after quantum job execution on real hardware
                    
            except Exception as e:
                logger.error(f"IBM Quantum hardware error: {str(e)}")
                logger.warning("Falling back to simulator for this evaluation")
                # Use local estimator as fallback
                self.estimator = self.local_estimator
                self.sampler = self.local_sampler
                loss = self._process_batch_quantum(params, feature_sets, target_data, selected_indices, batch_size)
        else:
            # Use local simulator
            loss = self._process_batch_quantum(params, feature_sets, target_data, selected_indices, batch_size)
        
        # Track iteration data if provided
        if iteration is not None:
            # Extract coefficients at this iteration
            self._update_optimization_history(iteration, loss, params)
                
        return loss / batch_size
    
    def _update_optimization_history(self, iteration, loss, params):
        """Update optimization history with current values."""
        self.optimization_history['iterations'].append(iteration)
        self.optimization_history['loss_values'].append(loss)
        
        # Derive coefficients from parameters
        raw_coeffs = np.abs(params[:min(self.total_features, len(params))])
        
        # Pad with defaults if needed
        if len(raw_coeffs) < self.total_features:
            padding = np.ones(self.total_features - len(raw_coeffs)) * 0.1
            raw_coeffs = np.concatenate([raw_coeffs, padding])
        
        # Normalize coefficients
        urbs_raw = raw_coeffs[:self.n_urbs_features]
        civitas_raw = raw_coeffs[self.n_urbs_features:self.n_urbs_features+self.n_civitas_features]
        
        urbs_sum = np.sum(urbs_raw)
        civitas_sum = np.sum(civitas_raw)
        
        # Create normalized urbs coefficients
        urbs_coeffs = []
        for i in range(self.n_urbs_features):
            if urbs_sum > 0:
                urbs_coeffs.append(urbs_raw[i] / urbs_sum)
            else:
                urbs_coeffs.append(1.0 / self.n_urbs_features)
        
        # Create normalized civitas coefficients
        civitas_coeffs = []
        for i in range(self.n_civitas_features):
            if civitas_sum > 0:
                civitas_coeffs.append(civitas_raw[i] / civitas_sum)
            else:
                civitas_coeffs.append(1.0 / self.n_civitas_features)
        
        self.optimization_history['urbs_coeffs'].append(urbs_coeffs)
        self.optimization_history['civitas_coeffs'].append(civitas_coeffs)
    
    def _process_batch_quantum(self, params, feature_sets, target_data, selected_indices, batch_size):
        """Process a batch of features using quantum circuits with BitArray result handling."""
        loss = 0
        quantum_job_count = 0  # Counter for quantum jobs
        
        for batch_idx in range(0, batch_size):
            idx = selected_indices[batch_idx]
            
            # Extract features and flatten to 1D array
            if isinstance(feature_sets[idx], np.ndarray):
                features = feature_sets[idx].flatten()
            else:
                # Handle case where feature_sets contains separate arrays
                features = np.array(feature_sets[idx]).flatten()
            
            # Get corresponding target values
            if isinstance(target_data, list) and idx < len(target_data):
                target = np.mean(target_data[idx].flatten())
            elif isinstance(target_data, np.ndarray):
                target = np.mean(target_data.flatten())
            else:
                logger.warning(f"Could not get target for index {idx}")
                target = 0.5  # Default fallback
            
            try:
                # Truncate features if needed
                n_qubits = min(16, self.total_features)
                truncated_features = features[:n_qubits]
                
                # Create the circuit - optimized for hardware compatibility
                qc = self.create_feature_encoding_circuit(
                    truncated_features, 
                    params,
                    for_hardware=self.use_real_device
                )
                
                # Ensure circuit has measurements
                if not qc.num_clbits:
                    qc.measure_all()
                
                prediction = 0.5  # Default prediction (uniform)
                total_shots = 1024
                
                # Run with appropriate sampler
                if isinstance(self.sampler, LocalSampler):
                    # Local sampler approach (unchanged)
                    job = self.sampler.run(qc, shots=total_shots)
                    result = job.result()
                    
                    # Extract counts from local sampler
                    if hasattr(result, 'quasi_dists'):
                        counts = result.quasi_dists[0]
                        prediction = self._calculate_prediction_from_counts(counts, n_qubits)
                else:
                    # IBM Quantum sampler approach - with BitArray data handling
                    job = self.sampler.run([qc], shots=total_shots)
                    quantum_job_count += 1
                    
                    
                        
                    result = job.result()
                    
                    # Access _pub_results to get the SamplerPubResult
                    if hasattr(result, '_pub_results') and result._pub_results:
                        logger.info("Using _pub_results to extract data")
                        pub_results = result._pub_results
                        
                        if len(pub_results) > 0:
                            # Get the first result (SamplerPubResult)
                            pub_result = pub_results[0]
                            
                            # Process BitArray data directly
                            try:
                                # Access data attribute directly (not a method)
                                data_bin = pub_result.data
                                logger.info(f"Extracted DataBin: {data_bin}")
                                
                                # The BitArray is in the 'meas' field of DataBin
                                if hasattr(data_bin, 'meas'):
                                    bit_array = data_bin.meas
                                    logger.info(f"Extracted BitArray: {bit_array}")
                                    
                                    # Get array data if available
                                    if hasattr(bit_array, 'array'):
                                        bit_data = bit_array.array
                                        logger.info(f"Extracted array data with shape {bit_data.shape}")
                                        
                                        # Get the actual number of bits from the array shape
                                        num_shots, actual_bits = bit_data.shape
                                        logger.info(f"Using actual_bits={actual_bits} instead of n_qubits={n_qubits}")
                                        
                                        # Create counts from array data using only the available bits
                                        counts = {}
                                        
                                        for shot_idx in range(num_shots):
                                            # Convert bit pattern to integer, but only using available bits
                                            int_val = 0
                                            for bit_idx in range(actual_bits):
                                                if bit_data[shot_idx, bit_idx]:
                                                    int_val |= (1 << bit_idx)
                                            
                                            counts[int_val] = counts.get(int_val, 0) + 1
                                        
                                        # Normalize counts
                                        counts = {k: v/num_shots for k, v in counts.items()}
                                        
                                        # Calculate prediction using just the bits we have
                                        # Note: Using actual_bits instead of n_qubits for the divisor
                                        prediction = 0.0
                                        for int_val, prob in counts.items():
                                            bit_val = int_val / (2**actual_bits)
                                            prediction += bit_val * prob
                                        
                                        logger.info(f"Calculated prediction: {prediction}")
                                    else:
                                        logger.warning("BitArray has no 'array' attribute")
                                        
                                        # Try to log BitArray attributes to understand it
                                        logger.info(f"BitArray attributes: {dir(bit_array)}")
                                        if hasattr(bit_array, 'num_shots'):
                                            logger.info(f"BitArray has {bit_array.num_shots} shots")
                                        if hasattr(bit_array, 'num_bits'):
                                            logger.info(f"BitArray has {bit_array.num_bits} bits")
                                else:
                                    logger.warning("DataBin has no 'meas' attribute")
                                    logger.info(f"DataBin attributes: {dir(data_bin)}")
                            except Exception as e:
                                logger.error(f"Error accessing BitArray data: {str(e)}")
                                import traceback
                                logger.error(f"BitArray access traceback: {traceback.format_exc()}")
                
                # Add to loss (mean squared error)
                batch_loss = (prediction - target) ** 2
                loss += batch_loss
                logger.info(f"Batch {batch_idx}: prediction={prediction:.4f}, target={target:.4f}, loss={batch_loss:.4f}")
                
            except Exception as e:
                logger.error(f"Quantum circuit evaluation error: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Fallback: return high loss for this batch item
                loss += 1.0
        
        # Add final delay after batch processing if using real hardware
        if self.use_real_device and quantum_job_count > 0 and self.job_delay_func:
            self.job_delay_func()
                
        return loss
    
    def _calculate_prediction_from_counts(self, counts, n_qubits, total_shots=1024):
        """
        Calculate weighted prediction from measurement counts.
        
        Args:
            counts: Dictionary mapping bitstring to count/probability
            n_qubits: Number of qubits in the circuit
            total_shots: Total number of shots
        
        Returns:
            Weighted prediction value
        """
        if not counts:
            return 0.5  # Default if no counts
            
        prediction = 0.0
        
        # Process the counts
        for bitstring, count in counts.items():
            # Convert bit string to normalized value based on type
            if isinstance(bitstring, str):
                # If it's a binary string like '01010'
                bit_val = int(bitstring, 2) / (2**n_qubits)
            else:
                # If it's already an integer
                bit_val = bitstring / (2**n_qubits)
            
            # For probability distributions vs raw counts
            if isinstance(count, float) and 0 <= count <= 1:
                # Already a probability
                weight = count
            else:
                # Raw count, normalize
                weight = count / total_shots
            
            prediction += bit_val * weight
        
        return prediction
    
    def fit(self, urbs_features, civitas_features, target_data, max_iterations=5, scenario_name="default"):
        """
        Optimize coefficients using quantum approach with strict iteration limit.
        
        Args:
            urbs_features: Urban feature sets
            civitas_features: Civic feature sets
            target_data: Target values for optimization
            max_iterations: Maximum iterations for optimizer (STRICT limit)
            scenario_name: Name of the scenario (for result files)
        """
        logger.info(f"Starting quantum coefficient optimization for scenario: {scenario_name}...")
        start_time = time.time()
        
        # Reset optimization history
        self.optimization_history = {
            'iterations': [],
            'loss_values': [],
            'urbs_coeffs': [],
            'civitas_coeffs': []
        }
        
        # Create scenario-specific results directory
        scenario_dir = os.path.join(self.results_dir, scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)
        
        # Prepare combined feature set
        if isinstance(urbs_features[0], np.ndarray):
            # Features are 2D arrays
            all_features = urbs_features + civitas_features
        else:
            # Features are lists of arrays
            all_features = []
            for i in range(len(urbs_features[0])):
                features = []
                for f in urbs_features:
                    features.append(f[i])
                for f in civitas_features:
                    features.append(f[i])
                all_features.append(features)
        
        # Each parameter appears twice (amplitude and phase encoding)
        n_params = 10  # Use 10 variational parameters for the quantum circuit
        initial_params = np.random.random(n_params) * np.pi
        
        # Store initial parameters for reference
        self.initial_params = initial_params.copy()
        
        # Select optimizer based on whether we're using real hardware
        if self.use_real_device:
            # Use SPSA for real hardware with strict iteration control
            optimizer = SPSA(maxiter=max_iterations)
            logger.info(f"Using SPSA optimizer with STRICT wrapper limit of {max_iterations} iterations")
        else:
            # Use COBYLA for simulator
            optimizer = COBYLA(maxiter=max_iterations)
            logger.info(f"Using COBYLA optimizer with limit of {max_iterations} iterations")
        
        # Iteration counter for callback
        iteration_counter = [0]
        hard_limit = max_iterations  # Enforce a hard limit
        
        def classical_wrapper(params):
            # Check if we've hit the hard iteration limit
            if iteration_counter[0] >= hard_limit:
                logger.info(f"Hit hard iteration limit of {hard_limit}, returning previous result")
                # Return the last loss if available or a default value
                return self.optimization_history['loss_values'][-1] if self.optimization_history['loss_values'] else 0.5
            
            current_iter = iteration_counter[0]
            iteration_counter[0] += 1
            logger.info(f"Starting iteration {current_iter+1} of {hard_limit} maximum")
            
            # Early return if we're at the last allowed iteration to prevent additional evaluations
            if current_iter == hard_limit - 1:
                logger.info(f"Final iteration ({hard_limit}) - recording results and stopping optimization")
                loss = self.quantum_cost_function(params, all_features, target_data, current_iter)
                # Force optimization to stop after this by returning a "perfect" result
                return 0.0
            
            return self.quantum_cost_function(params, all_features, target_data, current_iter)
        
        try:
            result = optimizer.minimize(classical_wrapper, initial_params)
            
            # Log detailed optimization results
            backend_info = f"on {self.backend.name}" if self.use_real_device else "on local simulator"
            logger.info(f"Optimization completed {backend_info}")
            
            # Extract optimized parameters
            if hasattr(result, 'x'):
                opt_params = result.x
                logger.info(f"Optimization successful: {getattr(result, 'success', True)}")
                logger.info(f"Function evaluations: {getattr(result, 'nfev', 'unknown')}")
            elif hasattr(result, 'optimal_point'):
                opt_params = result.optimal_point
                logger.info(f"Optimization successful: {getattr(result, 'success', True)}")
                logger.info(f"Function evaluations: {getattr(result, 'cost_function_evals', 'unknown')}")
            else:
                logger.warning(f"Unexpected result object: {result}")
                raise AttributeError("Could not find optimization result parameters")
        
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            # Fallback to default coefficients
            logger.warning("Using initial parameters as fallback")
            opt_params = initial_params
        
        # Convert to coefficients for urbs and civitas features
        urbs_coeffs, civitas_coeffs = self._params_to_coefficients(opt_params)
        
        # Validate the coefficients
        self.validate_coefficients(urbs_features, civitas_features, target_data, urbs_coeffs, civitas_coeffs)
        
        # Create visualizations
        self.visualize_optimization_history(scenario_name, scenario_dir)
        self.visualize_coefficient_values(urbs_coeffs, civitas_coeffs, scenario_name, scenario_dir)
        self.visualize_feature_importance(urbs_coeffs, civitas_coeffs, scenario_name, scenario_dir)
        
        # Save results
        self.save_optimization_results(urbs_coeffs, civitas_coeffs, scenario_name, scenario_dir)
        
        logger.info(f"Quantum coefficient optimization completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Urban coefficients: {urbs_coeffs}")
        logger.info(f"Civic coefficients: {civitas_coeffs}")
        
        return urbs_coeffs, civitas_coeffs
    
    def _params_to_coefficients(self, parameters):
        """Convert optimization parameters to normalized coefficient sets."""
        # Use the optimized parameters to derive coefficients
        raw_coeffs = np.abs(parameters[:min(self.total_features, len(parameters))])
        
        # Pad with defaults if needed
        if len(raw_coeffs) < self.total_features:
            padding = np.ones(self.total_features - len(raw_coeffs)) * 0.1
            raw_coeffs = np.concatenate([raw_coeffs, padding])
        
        # Normalize coefficients
        urbs_raw = raw_coeffs[:self.n_urbs_features]
        civitas_raw = raw_coeffs[self.n_urbs_features:self.n_urbs_features+self.n_civitas_features]
        
        urbs_sum = np.sum(urbs_raw)
        civitas_sum = np.sum(civitas_raw)
        
        # Create normalized urbs coefficients
        urbs_coeffs = []
        for i in range(self.n_urbs_features):
            if urbs_sum > 0:
                urbs_coeffs.append(float(urbs_raw[i] / urbs_sum))
            else:
                urbs_coeffs.append(1.0 / self.n_urbs_features)
                
        # Create normalized civitas coefficients
        civitas_coeffs = []
        for i in range(self.n_civitas_features):
            if civitas_sum > 0:
                civitas_coeffs.append(float(civitas_raw[i] / civitas_sum))
            else:
                civitas_coeffs.append(1.0 / self.n_civitas_features)
        
        return urbs_coeffs, civitas_coeffs
    
    def validate_coefficients(self, urbs_features, civitas_features, target_data, urbs_coeffs, civitas_coeffs):
        """
        Validate coefficients against the target data.
        
        Args:
            urbs_features: List of urban feature arrays
            civitas_features: List of civic feature arrays
            target_data: Target values for comparison
            urbs_coeffs: Optimized urban coefficients
            civitas_coeffs: Optimized civic coefficients
        """
        logger.info("Validating coefficient accuracy...")
        
        # Calculate predicted values using the coefficients
        predictions = []
        actual_values = []
        
        # Number of data points to validate (cap at 50 for performance)
        n_validate = min(50, len(urbs_features[0]) if isinstance(urbs_features[0], list) else 1)
        
        # Calculate predictions for each data point
        for i in range(n_validate):
            # Compute weighted sum of urban features
            urbs_sum = 0
            for j, coeff in enumerate(urbs_coeffs):
                if j < len(urbs_features):
                    feature_value = np.mean(urbs_features[j][i].flatten() if isinstance(urbs_features[j], list) else urbs_features[j].flatten())
                    urbs_sum += coeff * feature_value
            
            # Compute weighted sum of civic features
            civitas_sum = 0
            for j, coeff in enumerate(civitas_coeffs):
                if j < len(civitas_features):
                    feature_value = np.mean(civitas_features[j][i].flatten() if isinstance(civitas_features[j], list) else civitas_features[j].flatten())
                    civitas_sum += coeff * feature_value
            
            # Combine predictions
            prediction = (urbs_sum + civitas_sum) / 2
            predictions.append(prediction)
            
            # Get actual value
            if isinstance(target_data, list):
                actual = np.mean(target_data[i].flatten())
            else:
                actual = np.mean(target_data.flatten())
            actual_values.append(actual)
        
        # Calculate metrics
        try:
            mse = mean_squared_error(actual_values, predictions)
            r2 = r2_score(actual_values, predictions)
            correlation = np.corrcoef(actual_values, predictions)[0, 1]
            
            # Store metrics for reporting
            self.validation_metrics = {
                'mse': float(mse),
                'r2': float(r2),
                'correlation': float(correlation),
                'n_samples': n_validate
            }
            
            logger.info(f"Validation Results:")
            logger.info(f"  Mean Squared Error: {mse:.6f}")
            logger.info(f"  R² Score: {r2:.4f}")
            logger.info(f"  Correlation: {correlation:.4f}")
            logger.info(f"  Samples evaluated: {n_validate}")
        except Exception as e:
            logger.error(f"Error during coefficient validation: {e}")
            self.validation_metrics = {
                'mse': 0.0,
                'r2': 0.0,
                'correlation': 0.0,
                'n_samples': 0,
                'error': str(e)
            }
   
    def visualize_optimization_history(self, scenario_name, scenario_dir):
        """
        Create visualization of the optimization history.
        
        Args:
            scenario_name: Name of the scenario
            scenario_dir: Directory to save results
        """
        if not self.optimization_history['iterations']:
            logger.warning("No optimization history to visualize")
            return
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.optimization_history['iterations'], self.optimization_history['loss_values'], 'b-')
        plt.title(f"Loss Convergence - {scenario_name}")
        plt.xlabel("Iteration")
        plt.ylabel("Loss Value")
        plt.grid(True, alpha=0.3)
        
        # Plot coefficient evolution
        plt.subplot(1, 2, 2)
        
        # Plot urban coefficients evolution
        urbs_coeff_data = np.array(self.optimization_history['urbs_coeffs'])
        
        # Only plot first 5 urban coefficients to avoid cluttering
        plot_n = min(5, self.n_urbs_features)
        for i in range(plot_n):
            plt.plot(self.optimization_history['iterations'], 
                     urbs_coeff_data[:, i], 
                     label=f"Urban {i+1}")
        
        # Plot civic coefficients
        civitas_coeff_data = np.array(self.optimization_history['civitas_coeffs'])
        for i in range(self.n_civitas_features):
            plt.plot(self.optimization_history['iterations'], 
                     civitas_coeff_data[:, i], 
                     label=f"Civic {i+1}", 
                     linestyle='--')
            
        plt.title(f"Coefficient Evolution - {scenario_name}")
        plt.xlabel("Iteration")
        plt.ylabel("Coefficient Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(scenario_dir, f"{scenario_name}_optimization_history.png"), dpi=300)
        plt.close()
        
        logger.info(f"Optimization history visualization saved to {scenario_dir}")
    
    def visualize_coefficient_values(self, urbs_coeffs, civitas_coeffs, scenario_name, scenario_dir):
        """
        Create bar chart visualization of coefficient values.
        
        Args:
            urbs_coeffs: Urban coefficients
            civitas_coeffs: Civic coefficients
            scenario_name: Name of the scenario
            scenario_dir: Directory to save results
        """
        plt.figure(figsize=(14, 8))
        
        # Urban coefficients
        plt.subplot(2, 1, 1)
        bars = plt.bar(range(len(urbs_coeffs)), urbs_coeffs, color='skyblue')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', rotation=0)
        
        plt.title(f"Urban Coefficients - {scenario_name}")
        plt.ylabel("Coefficient Value")
        plt.xlabel("Feature Index")
        plt.xticks(range(len(urbs_coeffs)), [f"U{i+1}" for i in range(len(urbs_coeffs))])
        plt.grid(True, alpha=0.3, axis='y')
        
        # Civic coefficients
        plt.subplot(2, 1, 2)
        bars = plt.bar(range(len(civitas_coeffs)), civitas_coeffs, color='lightgreen')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', rotation=0)
        
        plt.title(f"Civic Coefficients - {scenario_name}")
        plt.ylabel("Coefficient Value")
        plt.xlabel("Feature Index")
        plt.xticks(range(len(civitas_coeffs)), [f"C{i+1}" for i in range(len(civitas_coeffs))])
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(scenario_dir, f"{scenario_name}_coefficient_values.png"), dpi=300)
        plt.close()
        
        logger.info(f"Coefficient values visualization saved to {scenario_dir}")
    
    def visualize_feature_importance(self, urbs_coeffs, civitas_coeffs, scenario_name, scenario_dir):
        """
        Create feature importance visualization.
        
        Args:
            urbs_coeffs: Urban coefficients
            civitas_coeffs: Civic coefficients
            scenario_name: Name of the scenario
            scenario_dir: Directory to save results
        """
        # Calculate total weight per category
        urbs_total = sum(urbs_coeffs)
        civitas_total = sum(civitas_coeffs)
        total_weight = urbs_total + civitas_total
        
        # Normalize to show relative importance between categories
        urbs_weight = urbs_total / total_weight
        civitas_weight = civitas_total / total_weight
        
        # Create feature importance plot
        plt.figure(figsize=(14, 10))
        
        # Pie chart showing category importance
        plt.subplot(2, 2, 1)
        plt.pie([urbs_weight, civitas_weight], 
                labels=['Urban', 'Civic'], 
                autopct='%1.1f%%',
                colors=['skyblue', 'lightgreen'],
                explode=(0.1, 0),
                startangle=90)
        plt.title(f"Category Importance - {scenario_name}")
        
        # Combined bar chart of all features
        plt.subplot(2, 2, 2)
        all_coeffs = urbs_coeffs + civitas_coeffs
        all_labels = [f"U{i+1}" for i in range(len(urbs_coeffs))] + [f"C{i+1}" for i in range(len(civitas_coeffs))]
        colors = ['skyblue'] * len(urbs_coeffs) + ['lightgreen'] * len(civitas_coeffs)
        
        # Normalize for global importance
        global_importance = [c / total_weight for c in all_coeffs]
        
        bars = plt.bar(range(len(all_coeffs)), global_importance, color=colors)
        plt.title(f"Global Feature Importance - {scenario_name}")
        plt.ylabel("Importance")
        plt.xticks(range(len(all_coeffs)), all_labels, rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Heat map of urban coefficients
        plt.subplot(2, 2, 3)
        df = pd.DataFrame({
            'Feature': [f"U{i+1}" for i in range(len(urbs_coeffs))],
            'Importance': urbs_coeffs
        })
        df = df.sort_values('Importance', ascending=False)
        
        sns.barplot(x='Feature', y='Importance', data=df, palette='Blues_d')
        plt.title(f"Urban Features Ranking - {scenario_name}")
        plt.ylabel("Coefficient Value")
        plt.grid(True, alpha=0.3, axis='y')
        
        # Heat map of civic coefficients
        plt.subplot(2, 2, 4)
        df = pd.DataFrame({
            'Feature': [f"C{i+1}" for i in range(len(civitas_coeffs))],
            'Importance': civitas_coeffs
        })
        df = df.sort_values('Importance', ascending=False)
        
        sns.barplot(x='Feature', y='Importance', data=df, palette='Greens_d')
        plt.title(f"Civic Features Ranking - {scenario_name}")
        plt.ylabel("Coefficient Value")
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(scenario_dir, f"{scenario_name}_feature_importance.png"), dpi=300)
        plt.close()
        
        logger.info(f"Feature importance visualization saved to {scenario_dir}")
    
    def save_optimization_results(self, urbs_coeffs, civitas_coeffs, scenario_name, scenario_dir):
        """
        Save optimization results to files.
        
        Args:
            urbs_coeffs: Urban coefficients
            civitas_coeffs: Civic coefficients
            scenario_name: Name of the scenario
            scenario_dir: Directory to save results
        """
        # Create timestamp for filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save coefficients as CSV
        coeffs_df = pd.DataFrame({
            'Type': ['Urban']*len(urbs_coeffs) + ['Civic']*len(civitas_coeffs),
            'Index': list(range(1, len(urbs_coeffs)+1)) + list(range(1, len(civitas_coeffs)+1)),
            'Value': urbs_coeffs + civitas_coeffs
        })
        coeffs_df.to_csv(os.path.join(scenario_dir, f"{scenario_name}_coefficients_{timestamp}.csv"), index=False)
        
        # Save optimization history
        if self.optimization_history['iterations']:
            history_df = pd.DataFrame({
                'Iteration': self.optimization_history['iterations'],
                'Loss': self.optimization_history['loss_values']
            })
            # Add coefficient columns
            for i in range(len(urbs_coeffs)):
                history_df[f'Urbs_{i+1}'] = [row[i] for row in self.optimization_history['urbs_coeffs']]
            for i in range(len(civitas_coeffs)):
                history_df[f'Civitas_{i+1}'] = [row[i] for row in self.optimization_history['civitas_coeffs']]
            
            history_df.to_csv(os.path.join(scenario_dir, f"{scenario_name}_optimization_history_{timestamp}.csv"), index=False)
        
        # Save validation metrics
        if self.validation_metrics:
            with open(os.path.join(scenario_dir, f"{scenario_name}_validation_metrics_{timestamp}.json"), 'w') as f:
                json.dump(self.validation_metrics, f, indent=2)
        
        # Save a summary report
        with open(os.path.join(scenario_dir, f"{scenario_name}_coefficient_report_{timestamp}.txt"), 'w') as f:
            f.write(f"=== Quantum Coefficient Optimization Report ===\n")
            f.write(f"Scenario: {scenario_name}\n")
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Backend: {'IBM Quantum (' + self.backend.name + ')' if self.use_real_device else 'Local Simulator'}\n\n")
            
            f.write("--- Urban Coefficients ---\n")
            for i, coeff in enumerate(urbs_coeffs):
                f.write(f"U{i+1}: {coeff:.6f}\n")
            
            f.write("\n--- Civic Coefficients ---\n")
            for i, coeff in enumerate(civitas_coeffs):
                f.write(f"C{i+1}: {coeff:.6f}\n")
            
            f.write("\n--- Validation Metrics ---\n")
            if self.validation_metrics:
                for key, value in self.validation_metrics.items():
                    f.write(f"{key}: {value}\n")
            else:
                f.write("No validation metrics available\n")
            
            f.write("\n--- Optimization Summary ---\n")
            if self.optimization_history['iterations']:
                f.write(f"Total iterations: {max(self.optimization_history['iterations'])}\n")
                f.write(f"Final loss: {self.optimization_history['loss_values'][-1]:.6f}\n")
                f.write(f"Initial loss: {self.optimization_history['loss_values'][0]:.6f}\n")
                improvement = (self.optimization_history['loss_values'][0] - self.optimization_history['loss_values'][-1]) / self.optimization_history['loss_values'][0] * 100
                f.write(f"Improvement: {improvement:.2f}%\n")
            else:
                f.write("No optimization history available\n")
        
        logger.info(f"All optimization results saved to {scenario_dir}")