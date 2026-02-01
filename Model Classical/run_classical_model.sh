#!/bin/bash

# Run the classical pedestrian model

# Create directories if they don't exist
mkdir -p data outputs

echo "Running Classical Pedestrian Model"
python run_classical_model.py --data_path data --output_dir outputs --enhancement 1.5 --plot_factors

echo "Done! Results are available in the outputs directory."
echo "To compare with quantum model results (if available), run:"
echo "python compare_with_quantum.py --classical_dir outputs --quantum_dir quantum_outputs --output_dir comparison"