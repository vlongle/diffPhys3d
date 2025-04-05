import argparse
import os
import numpy as np
import open3d as o3d
import torch

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from f3rm_robot.load import load_nerfstudio_objaverse_outputs
from f3rm_robot.initial_proposals import dense_voxel_grid, density_threshold_mask, remove_statistical_outliers, voxel_downsample
from f3rm_robot.optimize import filter_gray_background, remove_floating_clusters, get_qp_feats, get_alpha

## NOTE: TODO: need to weight the voxel_feat by the alpha compositing eqn.

def extract_clip_voxel_grid(
    scene_path: str,
    output_path: str,
    min_bounds: Tuple[float, float, float] = (-0.5, -0.5, -0.5),
    max_bounds: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    voxel_size: float = 0.01,
    batch_size: int = 4096,
    alpha_weighted: bool = True,
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
        alpha_weighted: Whether to weight features by alpha (density)
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
    print(f"Voxel grid shape: {voxel_grid.shape}")
    
    # Get the original shape directly from the voxel grid
    original_shape = voxel_grid.shape[:-1]  # Exclude the last dimension (3 for xyz)
    print(f"Original shape: {original_shape}")
    
    # Flatten the voxel grid for processing
    voxel_grid_flat = voxel_grid.reshape(-1, 3)
    print(f"Voxel grid flat shape: {voxel_grid_flat.shape}")
    
    # Move voxel grid to device
    voxel_grid_flat = torch.tensor(voxel_grid_flat, dtype=torch.float32, device=device)
    
    # Get a sample output to determine feature dimension
    with torch.no_grad():
        sample_output = feature_field(voxel_grid_flat[:1])
        feature_dim = sample_output["feature"].shape[-1]
    
    print(f"Feature dimension: {feature_dim}")
    
    # Initialize arrays to store features, alpha values, and RGB values
    total_voxels = voxel_grid_flat.shape[0]
    print(f"Total voxels: {total_voxels}")
    
    # Use float16 to save memory
    features_cpu = np.zeros((total_voxels, feature_dim), dtype=np.float16)
    alphas_cpu = np.zeros((total_voxels, 1), dtype=np.float16)
    rgb_cpu = np.zeros((total_voxels, 3), dtype=np.float16)
    
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
            
            # Get features - either raw or alpha-weighted
            if alpha_weighted:
                # Weight features by alpha (density) as in get_qp_feats
                feature = get_qp_feats(outputs)
                print(f"Using alpha-weighted features (batch {i})" if i == 0 else "", end="\r")
            else:
                feature = outputs["feature"]
                print(f"Using raw features (batch {i})" if i == 0 else "", end="\r")
            
            # Get RGB values
            rgb = feature_field.get_rgb(batch)
            
            # Move to CPU and convert to float16 to save memory
            features_cpu[i:end_idx] = feature.cpu().to(torch.float16).numpy()
            alphas_cpu[i:end_idx] = alpha.cpu().to(torch.float16).numpy()
            rgb_cpu[i:end_idx] = rgb.cpu().to(torch.float16).numpy()
            
            # Free up GPU memory
            del outputs, alpha, feature, rgb
            torch.cuda.empty_cache()
    
    print(f"\nFeatures shape: {features_cpu.shape}")
    print(f"RGB shape: {rgb_cpu.shape}")
    
    # Reshape to original grid shape
    print("Reshaping arrays to grid format...")
    features_reshaped = features_cpu.reshape(*original_shape, feature_dim)
    alphas_reshaped = alphas_cpu.reshape(*original_shape, 1)
    rgb_reshaped = rgb_cpu.reshape(*original_shape, 3)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save metadata
    output_data = {
        "min_bounds": min_bounds,
        "max_bounds": max_bounds,
        "voxel_size": voxel_size,
        "feature_dim": feature_dim,
        "grid_shape": original_shape,
        "alpha_weighted": alpha_weighted,
    }
    
    # Save metadata and arrays separately
    np.savez_compressed(output_path, **output_data)
    
    # Save large arrays to separate files
    features_path = output_path.replace('.npz', '_features.npy')
    alphas_path = output_path.replace('.npz', '_alphas.npy')
    rgb_path = output_path.replace('.npz', '_rgb.npy')
    
    print(f"Saving features to {features_path}...")
    np.save(features_path, features_reshaped)
    
    print(f"Saving alphas to {alphas_path}...")
    np.save(alphas_path, alphas_reshaped)
    
    print(f"Saving RGB to {rgb_path}...")
    np.save(rgb_path, rgb_reshaped)
    
    print("Done!")
    return output_path


def compute_occupancy_point_cloud(
    feature_grid_path: str,
    alpha_threshold: float = 0.01,
    gray_threshold: float = 0.05,
    voxel_downsample_size: float = 0.01,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Compute an occupancy point cloud from saved voxel grid features and alphas.
    
    Args:
        feature_grid_path: Path to the saved feature grid metadata (.npz file)
        alpha_threshold: Threshold for density values to consider a voxel occupied
        gray_threshold: Threshold for detecting black background
        voxel_downsample_size: Size for downsampling the resulting point cloud
        device: Device to use for computation
        
    Returns:
        o3d.geometry.PointCloud: Filtered point cloud representing the scene
    """
    import open3d as o3d
    import torch
    
    print(f"Loading feature grid from {feature_grid_path}...")
    
    # Load metadata
    metadata = np.load(feature_grid_path)
    min_bounds = metadata['min_bounds']
    max_bounds = metadata['max_bounds']
    voxel_size = float(metadata['voxel_size'])
    grid_shape = metadata['grid_shape']
    
    print(f"Grid shape: {grid_shape}, voxel size: {voxel_size}")
    print(f"Bounds: min={min_bounds}, max={max_bounds}")
    
    # Load alphas
    alphas_path = feature_grid_path.replace('.npz', '_alphas.npy')
    print(f"Loading alphas from {alphas_path}...")
    alphas = np.load(alphas_path)
    
    # Load RGB
    rgb_path = feature_grid_path.replace('.npz', '_rgb.npy')
    print(f"Loading RGB from {rgb_path}...")
    rgb = np.load(rgb_path)
    
    # Convert to torch tensors
    alphas_tensor = torch.from_numpy(alphas).to(device)
    rgb_tensor = torch.from_numpy(rgb).to(device)
    
    # Create coordinate grid
    print("Creating coordinate grid...")
    x = torch.linspace(min_bounds[0], max_bounds[0], grid_shape[0], device=device)
    y = torch.linspace(min_bounds[1], max_bounds[1], grid_shape[1], device=device)
    z = torch.linspace(min_bounds[2], max_bounds[2], grid_shape[2], device=device)
    
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)
    
    # Flatten everything for processing
    coords_flat = coords.reshape(-1, 3)
    alphas_flat = alphas_tensor.reshape(-1, 1)
    rgb_flat = rgb_tensor.reshape(-1, 3)
    
    # Apply density thresholding
    print(f"Applying density threshold {alpha_threshold}...")
    density_mask = alphas_flat.squeeze(-1) > alpha_threshold
    coords_filtered = coords_flat[density_mask]
    rgb_filtered = rgb_flat[density_mask]
    
    print(f"After density filtering: {coords_filtered.shape[0]} points")
    
    # Create a mock feature field adapter for black background filtering
    class MockFeatureFieldAdapter:
        def get_rgb(self, points):
            return rgb_filtered
    
    mock_feature_field = MockFeatureFieldAdapter()
    
    # Apply black background filtering using the centralized function
    print(f"Applying black background filtering with threshold {gray_threshold}...")
    non_bg_mask = filter_gray_background(coords_filtered, mock_feature_field, gray_threshold, device,
                                         return_mask=True)
    
    # If filter_gray_background returns the filtered points directly
    if isinstance(non_bg_mask, torch.Tensor) and non_bg_mask.shape == coords_filtered.shape:
        coords_filtered = non_bg_mask
        # We need to get the RGB values for these filtered points
        with torch.no_grad():
            rgb_filtered = mock_feature_field.get_rgb(coords_filtered)
    else:
        # If it returns a mask
        coords_filtered = coords_filtered[non_bg_mask]
        rgb_filtered = rgb_filtered[non_bg_mask]
    
    print(f"After black background filtering: {coords_filtered.shape[0]} points")
    
    # Create Open3D point cloud
    print("Creating point cloud...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords_filtered.cpu().numpy())
    
    # Add colors
    # Normalize RGB values to [0, 1] if needed
    if rgb_filtered.max() > 1.0:
        rgb_filtered = rgb_filtered / 255.0
    pcd.colors = o3d.utility.Vector3dVector(rgb_filtered.cpu().numpy())
    
    # Downsample point cloud
    print(f"Downsampling with voxel size {voxel_downsample_size}...")
    pcd = pcd.voxel_down_sample(voxel_size=voxel_downsample_size)
    
    # Remove statistical outliers
    print("Removing outliers...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=4.0)
    
    # Remove floating clusters
    print("Removing floating clusters...")
    pcd = remove_floating_clusters(pcd, min_points=10, eps=voxel_downsample_size*5)
    
    print(f"Final point cloud has {len(pcd.points)} points")
    return pcd


def save_point_cloud(pcd, output_path):
    """
    Save a point cloud to a file.
    
    Args:
        pcd: Open3D point cloud
        output_path: Path to save the point cloud
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the point cloud
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved point cloud to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract CLIP features in voxel grid format from a trained F3RM model")
    parser.add_argument("--scene", type=str, required=True, help="Path to the trained F3RM model")
    parser.add_argument("--output", type=str, required=True, help="Path to save the extracted feature grid")
    parser.add_argument("--min_x", type=float, default=-0.5, help="Minimum x bound")
    parser.add_argument("--min_y", type=float, default=-0.5, help="Minimum y bound")
    parser.add_argument("--min_z", type=float, default=-0.5, help="Minimum z bound")
    parser.add_argument("--max_x", type=float, default=0.5, help="Maximum x bound")
    parser.add_argument("--max_y", type=float, default=0.5, help="Maximum y bound")
    parser.add_argument("--max_z", type=float, default=0.5, help="Maximum z bound")
    parser.add_argument("--voxel_size", type=float, default=0.01, help="Size of each voxel")
    parser.add_argument("--batch_size", type=int, default=4096, help="Number of voxels to process at once")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--alpha_weighted", action="store_true", default=True, help="Weight features by alpha (density)")
    
    # Add arguments for point cloud extraction
    parser.add_argument("--alpha_threshold", type=float, default=0.01, help="Threshold for density values")
    parser.add_argument("--gray_threshold", type=float, default=0.05, help="Threshold for black background filtering")
    parser.add_argument("--pc_output", type=str, help="Path to save the extracted point cloud")
    
    args = parser.parse_args()
    
    min_bounds = (args.min_x, args.min_y, args.min_z)
    max_bounds = (args.max_x, args.max_y, args.max_z)
    
    
    output_path = extract_clip_voxel_grid(
            args.scene,
            args.output,
            min_bounds,
            max_bounds,
            args.voxel_size,
            args.batch_size,
            args.alpha_weighted,
            args.device
        )
    print("Extracting point cloud from saved voxel grid...")
    pcd = compute_occupancy_point_cloud(
        output_path,
        alpha_threshold=args.alpha_threshold,
        gray_threshold=args.gray_threshold,
        voxel_downsample_size=args.voxel_size,
        device=args.device
    )
    
    # Save point cloud
    pc_output = args.pc_output or output_path.replace('.npz', '_pc.ply')
    save_point_cloud(pcd, pc_output)


if __name__ == "__main__":
    main()