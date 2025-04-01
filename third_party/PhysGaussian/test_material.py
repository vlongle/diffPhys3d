import json
import os
import subprocess
import shutil
from pathlib import Path

# Define the materials to test
materials = ["jelly", "metal", "sand", "foam", "snow", "plasticine"]

# Path to the original config file
original_config_path = "./config/custom_config.json"

# Load the original config
with open(original_config_path, 'r') as f:
    original_config = json.load(f)

# Path to the point cloud file - adjust if needed
point_cloud_path = "/home/vlongle/code/diffPhys3d/render_outputs/ecb91f433f144a7798724890f0528b23/clip_features_pc.ply"

# Create a directory for the configs if it doesn't exist
config_dir = Path("./material_configs")
config_dir.mkdir(exist_ok=True)

# Create a directory for the outputs if it doesn't exist
output_base_dir = Path("./material_outputs")
output_base_dir.mkdir(exist_ok=True)

# Function to run the simulation for a given material
def run_simulation(material):
    # Create a copy of the original config
    config = original_config.copy()
    
    # Update the material
    config["material"] = material
    
    # Adjust other parameters based on material if needed
    if material == "metal":
        config["E"] = 1e7  # Higher Young's modulus for metal
        config["nu"] = 0.3  # Different Poisson ratio
    elif material == "sand":
        config["E"] = 1e5  # Lower Young's modulus for sand
        config["friction_angle"] = 30.0  # Add friction angle for sand
    elif material == "foam":
        config["E"] = 5e5  # Moderate Young's modulus for foam
    elif material == "snow":
        config["E"] = 1e5  # Lower Young's modulus for snow
        config["softening"] = 0.2  # More softening for snow
    elif material == "plasticine":
        config["E"] = 8e5  # Moderate Young's modulus for plasticine
        config["yield_stress"] = 1e4  # Add yield stress for plasticine
    
    # Save the modified config
    config_path = config_dir / f"config_{material}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Define the output path
    output_path = f"nerf_pc_{material}_output"
    
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
    
    print(f"Running simulation for {material}...")
    print(" ".join(cmd))
    
    subprocess.run(cmd, check=True)
    
    print(f"Completed simulation for {material}\n")

# Run simulations for all materials
for material in materials:
    run_simulation(material)

print("All simulations completed!")
