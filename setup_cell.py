import os

# Define local paths
base_dir = r"D:\DATA\MALITH\Uni\Semester 08\ISRP\Model GIS\Model_VS_Qiskit"
inputs_dir = os.path.join(base_dir, 'inputs')
results_dir = os.path.join(base_dir, 'results')

# Create project directory structure
os.makedirs(inputs_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Change working directory to project folder
os.chdir(base_dir)

print("Setup complete!")
print(f"Your data should be in '{inputs_dir}'")
print(f"Results will be saved to '{results_dir}'")

# Verify Qiskit installation
try:
    import qiskit
    print(f"\nQiskit version: {qiskit.__version__}")
except ImportError:
    print("Qiskit is not installed. Run 'pip install qiskit' to install it.")

# Show available Qiskit packages
import subprocess
subprocess.run("pip list | findstr qiskit", shell=True)
