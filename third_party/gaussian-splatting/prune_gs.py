import os
import numpy as np
import argparse
from plyfile import PlyData, PlyElement
import shutil

def prune_gaussians_outside_unit_cube(model_path, output_path, iteration=None, cube_size=[1, 1, 1]):
    """
    Prune Gaussians outside a cube centered at origin with customizable size.
    
    Args:
        model_path: Path to the trained model
        output_path: Path to save the pruned model
        iteration: Specific iteration to load, if None loads the latest
        cube_size: Size of the cube [x, y, z], default is [1, 1, 1]
    """
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "point_cloud"), exist_ok=True)
    
    # Determine which iteration to use
    if iteration is None or iteration == -1:
        # Find the latest iteration
        iterations_dir = os.path.join(model_path, "point_cloud")
        iterations = [d for d in os.listdir(iterations_dir) if d.startswith("iteration_")]
        iterations.sort(key=lambda x: int(x.split("_")[1]))
        if not iterations:
            raise ValueError(f"No iterations found in {iterations_dir}")
        iteration_dir = iterations[-1]
    else:
        iteration_dir = f"iteration_{iteration}"
    
    # Source PLY file path
    ply_path = os.path.join(model_path, "point_cloud", iteration_dir, "point_cloud.ply")
    
    # Load the PLY file
    print(f"Loading PLY file from {ply_path}")
    ply_data = PlyData.read(ply_path)
    
    # Extract vertex data
    vertices = ply_data['vertex']
    
    # Get positions
    x = vertices['x']
    y = vertices['y']
    z = vertices['z']
    
    # Calculate half-sizes for the cube
    half_size_x = cube_size[0] / 2
    half_size_y = cube_size[1] / 2
    half_size_z = cube_size[2] / 2
    
    # Create mask for points inside the cube
    inside_mask = (x >= -half_size_x) & (x <= half_size_x) & \
                  (y >= -half_size_y) & (y <= half_size_y) & \
                  (z >= -half_size_z) & (z <= half_size_z)
    
    # Print statistics
    total_gaussians = len(x)
    remaining_gaussians = np.sum(inside_mask)
    pruned_gaussians = total_gaussians - remaining_gaussians
    
    print(f"Total Gaussians: {total_gaussians}")
    print(f"Gaussians outside cube of size {cube_size}: {pruned_gaussians}")
    print(f"Remaining Gaussians: {remaining_gaussians}")
    
    # Filter the vertices
    filtered_vertices = vertices[inside_mask]
    
    # Create a new PLY file with filtered vertices
    new_ply = PlyData([PlyElement.describe(filtered_vertices, 'vertex')], text=True)
    
    os.makedirs(os.path.join(output_path, f"point_cloud/{iteration_dir}"), exist_ok=True)

    # Output PLY path
    output_ply_path = os.path.join(output_path, f"point_cloud/{iteration_dir}/point_cloud.ply")
    
    # Save the new PLY file
    new_ply.write(output_ply_path)
    print(f"Saved pruned point cloud to {output_ply_path}")
    
    # Copy other necessary files
    # Copy cameras.json if it exists
    src_cameras_path = os.path.join(model_path, "cameras.json")
    dst_cameras_path = os.path.join(output_path, "cameras.json")
    if os.path.exists(src_cameras_path):
        shutil.copy2(src_cameras_path, dst_cameras_path)
        print(f"Copied camera data to {dst_cameras_path}")
    
    # Copy exposure.json if it exists
    src_exposure_path = os.path.join(model_path, "exposure.json")
    dst_exposure_path = os.path.join(output_path, "exposure.json")
    if os.path.exists(src_exposure_path):
        shutil.copy2(src_exposure_path, dst_exposure_path)
        print(f"Copied exposure data to {dst_exposure_path}")
    
    # Copy scene metadata if it exists
    src_metadata_path = os.path.join(model_path, "scene_metadata.json")
    dst_metadata_path = os.path.join(output_path, "scene_metadata.json")
    if os.path.exists(src_metadata_path):
        shutil.copy2(src_metadata_path, dst_metadata_path)
        print(f"Copied scene metadata to {dst_metadata_path}")
    
    # Copy config args if it exists
    src_cfg_path = os.path.join(model_path, "cfg_args")
    dst_cfg_path = os.path.join(output_path, "cfg_args")
    if os.path.exists(src_cfg_path):
        shutil.copy2(src_cfg_path, dst_cfg_path)
        print(f"Copied config args to {dst_cfg_path}")
    
    return remaining_gaussians

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune Gaussians outside a cube")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the pruned model")
    parser.add_argument("--iteration", type=int, default=-1, help="Specific iteration to load (default: latest)")
    parser.add_argument("--cube_size", type=float, nargs=3, default=[1, 1, 1], 
                        help="Size of the cube [x, y, z] (default: [1, 1, 1])")
    
    args = parser.parse_args()
    
    prune_gaussians_outside_unit_cube(args.model_path, args.output_path, args.iteration, args.cube_size)
