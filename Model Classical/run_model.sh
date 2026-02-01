#!/bin/bash

# Script to run the Classical Pedestrian Model

# Create necessary directories if they don't exist
mkdir -p data outputs

echo "Running Classical Pedestrian Model"

# Generate synthetic data if needed
echo "Generating synthetic data for testing..."
python example_usage.py

echo "Running the full model with validation..."
python -c "from classical_model import ClassicalPedestrianModel; \
           import rasterio; \
           model = ClassicalPedestrianModel(); \
           observed = rasterio.open('data/observed_pedestrian.tif').read(1); \
           model.run_model('data/v_urbs.tif', 'data/v_civitas.tif', observed)"

echo "Model execution completed successfully."
echo "Results are available in the 'outputs' directory."

# Display summary of results
echo ""
echo "Summary of Results:"
echo "-------------------"
if [ -f "outputs/validation_metrics.txt" ]; then
    cat outputs/validation_metrics.txt
else
    echo "See pedestrian_probability.tif for the final probability map"
    echo "See pedestrian_probability.png for the visualization"
fi