import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from f3rm_robot.load import load_nerfstudio_objaverse_outputs
from f3rm_robot.initial_proposals import dense_voxel_grid


def extract_occupancy_voxel_grid(
    scene_path: str,
    output_path: str,
    min_bounds: Tuple[float, float, float] = (-0.5, -0.5, -0.5),
    max_bounds: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    voxel_size: float = 0.01,
    alpha_threshold: float = 0.1,
    batch_size: int = 4096,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Extract occupancy voxel grid from a trained F3RM model.
    
    Args:
        scene_path: Path to the trained F3RM model
        output_path: Path to save the extracted occupancy grid
        min_bounds: Minimum bounds of the voxel grid in world coordinates
        max_bounds: Maximum bounds of the voxel grid in world coordinates
        voxel_size: Size of each voxel
        alpha_threshold: Threshold for considering a voxel as occupied
        batch_size: Number of voxels to process at once
        device: Device to use for computation
    """
    # Load the feature field
    print(f"Loading feature field from {scene_path}...")
    load_state = load_nerfstudio_objaverse_outputs(scene_path)
    feature_field = load_state.feature_field_adapter()
    
    # Get the nerf_to_world transform matrix if available
    nerf_to_world = load_state.nerf_to_world
    world_to_nerf = None
    if hasattr(nerf_to_world, 'get_matrix'):
        # If it's a Transform3d object, get the matrix
        nerf_to_world_matrix = nerf_to_world.get_matrix()
        world_to_nerf_matrix = torch.inverse(nerf_to_world_matrix)
        
        # Save the transforms for rendering
        transform_path = output_path.replace('.npz', '_transforms.npz')
        np.savez_compressed(transform_path, 
                            nerf_to_world=nerf_to_world_matrix.cpu().numpy(),
                            world_to_nerf=world_to_nerf_matrix.cpu().numpy())
        print(f"Saved transforms to {transform_path}")
    
    # Create a dense voxel grid
    print(f"Creating voxel grid with voxel size {voxel_size}...")
    voxel_grid = dense_voxel_grid(
        min_bounds=min_bounds,
        max_bounds=max_bounds,
        voxel_size=voxel_size
    )
    
    # Calculate the original shape from the voxel grid
    original_shape = []
    for min_bound, max_bound in zip(min_bounds, max_bounds):
        dim_size = int((max_bound - min_bound) / voxel_size)
        original_shape.append(dim_size)
    original_shape = tuple(original_shape)
    
    # Flatten the voxel grid for processing
    voxel_grid_flat = voxel_grid.reshape(-1, 3)
    
    # Move voxel grid to device
    voxel_grid_flat = torch.tensor(voxel_grid_flat, dtype=torch.float32, device=device)
    
    # Initialize array to store alpha values (occupancy)
    total_voxels = voxel_grid_flat.shape[0]
    print(f"Total voxels: {total_voxels}")
    
    # Use boolean array to save memory for occupancy
    occupancy_cpu = np.zeros(total_voxels, dtype=bool)
    
    # Extract alpha values in batches
    print("Extracting occupancy values...")
    
    with torch.no_grad():
        for i in tqdm(range(0, total_voxels, batch_size)):
            end_idx = min(i + batch_size, total_voxels)
            batch = voxel_grid_flat[i:end_idx]
            
            # Get alpha values (density)
            alpha = feature_field.get_alpha(batch, voxel_size)
            
            # Determine occupancy based on alpha threshold
            occupancy = (alpha >= alpha_threshold).squeeze(-1)
            
            # Move to CPU and store
            occupancy_cpu[i:end_idx] = occupancy.cpu().numpy()
            
            # Free up GPU memory
            del alpha, occupancy
            torch.cuda.empty_cache()
    
    # Reshape to original grid shape
    print("Reshaping array to grid format...")
    occupancy_reshaped = occupancy_cpu.reshape(*original_shape)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the occupancy grid
    print(f"Saving occupancy grid to {output_path}...")
    
    # Save metadata and occupancy grid
    output_data = {
        "min_bounds": min_bounds,
        "max_bounds": max_bounds,
        "voxel_size": voxel_size,
        "alpha_threshold": alpha_threshold,
        "grid_shape": original_shape,
        "occupancy": occupancy_reshaped
    }
    
    np.savez_compressed(output_path, **output_data)
    
    # Calculate statistics
    num_occupied = np.sum(occupancy_reshaped)
    occupancy_percentage = (num_occupied / total_voxels) * 100
    print(f"Occupancy statistics:")
    print(f"  - Total voxels: {total_voxels}")
    print(f"  - Occupied voxels: {num_occupied}")
    print(f"  - Occupancy percentage: {occupancy_percentage:.2f}%")
    
    print("Done!")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Extract occupancy voxel grid from a trained F3RM model")
    parser.add_argument("--scene", type=str, required=True, help="Path to the trained F3RM model")
    parser.add_argument("--output", type=str, required=True, help="Path to save the extracted occupancy grid")
    parser.add_argument("--min_x", type=float, default=-0.5, help="Minimum x bound")
    parser.add_argument("--min_y", type=float, default=-0.5, help="Minimum y bound")
    parser.add_argument("--min_z", type=float, default=-0.5, help="Minimum z bound")
    parser.add_argument("--max_x", type=float, default=0.5, help="Maximum x bound")
    parser.add_argument("--max_y", type=float, default=0.5, help="Maximum y bound")
    parser.add_argument("--max_z", type=float, default=0.5, help="Maximum z bound")
    parser.add_argument("--voxel_size", type=float, default=0.01, help="Size of each voxel")
    parser.add_argument("--alpha_threshold", type=float, default=0.1, help="Threshold for considering a voxel as occupied")
    parser.add_argument("--batch_size", type=int, default=4096, help="Number of voxels to process at once")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    
    min_bounds = (args.min_x, args.min_y, args.min_z)
    max_bounds = (args.max_x, args.max_y, args.max_z)
    
    extract_occupancy_voxel_grid(
        args.scene,
        args.output,
        min_bounds,
        max_bounds,
        args.voxel_size,
        args.alpha_threshold,
        args.batch_size,
        args.device
    )


if __name__ == "__main__":
    main()