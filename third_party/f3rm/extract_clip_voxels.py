import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from f3rm_robot.load import load_nerfstudio_objaverse_outputs
from f3rm_robot.initial_proposals import dense_voxel_grid


def extract_clip_voxel_grid(
    scene_path: str,
    output_path: str,
    min_bounds: Tuple[float, float, float] = (-0.5, -0.5, -0.5),
    max_bounds: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    voxel_size: float = 0.01,
    batch_size: int = 4096,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Extract CLIP features in a voxel grid format from a trained F3RM model.
    
    Args:
        scene_path: Path to the trained F3RM model
        output_path: Path to save the extracted feature grid
        min_bounds: Minimum bounds of the voxel grid in world coordinates
        max_bounds: Maximum bounds of the voxel grid in world coordinates
        voxel_size: Size of each voxel
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
    
    # Get a sample output to determine feature dimension
    with torch.no_grad():
        sample_output = feature_field(voxel_grid_flat[:1])
        feature_dim = sample_output["feature"].shape[-1]
    
    print(f"Feature dimension: {feature_dim}")
    
    # Initialize arrays to store features and alpha values
    total_voxels = voxel_grid_flat.shape[0]
    print(f"Total voxels: {total_voxels}")
    
    # Use float16 to save memory
    features_cpu = np.zeros((total_voxels, feature_dim), dtype=np.float16)
    alphas_cpu = np.zeros((total_voxels, 1), dtype=np.float16)
    
    # Extract features in batches
    print("Extracting features...")
    
    with torch.no_grad():
        for i in tqdm(range(0, total_voxels, batch_size)):
            end_idx = min(i + batch_size, total_voxels)
            batch = voxel_grid_flat[i:end_idx]
            
            # Get outputs from feature field
            outputs = feature_field(batch)
            
            # Get alpha values (density)
            alpha = feature_field.get_alpha(batch, voxel_size)
            
            # Get features
            feature = outputs["feature"]
            
            # Move to CPU and convert to float16 to save memory
            features_cpu[i:end_idx] = feature.cpu().to(torch.float16).numpy()
            alphas_cpu[i:end_idx] = alpha.cpu().to(torch.float16).numpy()
            
            # Free up GPU memory
            del outputs, alpha, feature
            torch.cuda.empty_cache()
    
    # Reshape to original grid shape
    print("Reshaping arrays to grid format...")
    features_reshaped = features_cpu.reshape(*original_shape, feature_dim)
    alphas_reshaped = alphas_cpu.reshape(*original_shape, 1)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the feature grid and alpha values
    print(f"Saving feature grid to {output_path}...")
    
    # Save metadata
    output_data = {
        "min_bounds": min_bounds,
        "max_bounds": max_bounds,
        "voxel_size": voxel_size,
        "feature_dim": feature_dim,
        "grid_shape": original_shape,
    }
    
    # Save metadata and arrays separately
    np.savez_compressed(output_path, **output_data)
    
    # Save large arrays to separate files
    features_path = output_path.replace('.npz', '_features.npy')
    alphas_path = output_path.replace('.npz', '_alphas.npy')
    
    print(f"Saving features to {features_path}...")
    np.save(features_path, features_reshaped)
    
    print(f"Saving alphas to {alphas_path}...")
    np.save(alphas_path, alphas_reshaped)
    
    print("Done!")
    return output_path


### uhmmm, turns out the scene bounds are not very correct...

def main():
    parser = argparse.ArgumentParser(description="Extract CLIP features in voxel grid format from a trained F3RM model")
    parser.add_argument("--scene", type=str, required=True, help="Path to the trained F3RM model")
    parser.add_argument("--output", type=str, required=True, help="Path to save the extracted feature grid")
    # parser.add_argument("--min_x", type=float, default=-0.5, help="Minimum x bound")
    # parser.add_argument("--min_y", type=float, default=-0.5, help="Minimum y bound")
    # parser.add_argument("--min_z", type=float, default=-0.5, help="Minimum z bound")
    # parser.add_argument("--max_x", type=float, default=0.5, help="Maximum x bound")
    # parser.add_argument("--max_y", type=float, default=0.5, help="Maximum y bound")
    # parser.add_argument("--max_z", type=float, default=0.5, help="Maximum z bound")

    parser.add_argument("--min_x", type=float, default=-1.0, help="Minimum x bound")
    parser.add_argument("--min_y", type=float, default=-1.0, help="Minimum y bound")
    parser.add_argument("--min_z", type=float, default=-2.0, help="Minimum z bound")
    parser.add_argument("--max_x", type=float, default=1.0, help="Maximum x bound")
    parser.add_argument("--max_y", type=float, default=1.0, help="Maximum y bound")
    parser.add_argument("--max_z", type=float, default=0.25, help="Maximum z bound")

    parser.add_argument("--voxel_size", type=float, default=0.01, help="Size of each voxel")
    parser.add_argument("--batch_size", type=int, default=4096, help="Number of voxels to process at once")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    
    min_bounds = (args.min_x, args.min_y, args.min_z)
    max_bounds = (args.max_x, args.max_y, args.max_z)
    
    extract_clip_voxel_grid(
        args.scene,
        args.output,
        min_bounds,
        max_bounds,
        args.voxel_size,
        args.batch_size,
        args.device
    )


if __name__ == "__main__":
    main()