import json
import os
import subprocess
import shutil
from pathlib import Path

# Define the Young's modulus values to test
e_values = {"high_stiffness": 2e6, "low_stiffness": 2e4}

# Path to the original config file
original_config_path = "./config/custom_config.json"

# Load the original config
with open(original_config_path, 'r') as f:
    original_config = json.load(f)

# Path to the point cloud file - adjust if needed
point_cloud_path = "/home/vlongle/code/diffPhys3d/render_outputs/ecb91f433f144a7798724890f0528b23/clip_features_pc.ply"

# Create a directory for the configs if it doesn't exist
config_dir = Path("./property_configs")
config_dir.mkdir(exist_ok=True)

# Create a directory for the outputs if it doesn't exist
output_base_dir = Path("./property_outputs")
output_base_dir.mkdir(exist_ok=True)

# Function to run the simulation for a given Young's modulus value
def run_simulation(name, e_value):
    # Create a copy of the original config
    config = original_config.copy()
    
    # Update the Young's modulus
    config["E"] = e_value
    
    # Save the modified config
    config_path = config_dir / f"config_{name}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Define the output path
    output_path = f"nerf_pc_{name}_output"
    
    # Run the simulation command
    cmd = [
        "xvfb-run", "-a",
        "python", "gs_simulation_pc.py",
        "--point_cloud_path", point_cloud_path,
        "--output_path", output_path,
        "--config", str(config_path),
        "--render_img",
        "--compile_video",
        "--white_bg",
        "--debug"
    ]
    
    print(f"Running simulation for {name} (E = {e_value})...")
    print(" ".join(cmd))
    
    subprocess.run(cmd, check=True)
    
    print(f"Completed simulation for {name}\n")

# Run simulations for all Young's modulus values
for name, e_value in e_values.items():
    run_simulation(name, e_value)

print("All simulations completed!") 