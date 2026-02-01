import PyInstaller.__main__
import os
import sys
import shutil

# Define application name and main script
app_name = "QuantumUrbanExplorer"
main_script = "urban_scenario_simulator.py"

# Create a directory for output files
if os.path.exists("dist"):
    shutil.rmtree("dist")
os.makedirs("dist", exist_ok=True)

# Package the application without template references
PyInstaller.__main__.run([
    main_script,
    '--name=%s' % app_name,
    '--onefile',
    '--windowed',
    # Remove the references to non-existent directories
    # '--add-data=templates;templates',
    # '--add-data=static;static',
    '--icon=app_icon.ico' if os.path.exists('app_icon.ico') else '',
    '--clean',
])

print(f"Application packaged successfully. Executable is at dist/{app_name}.exe")