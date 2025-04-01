#!/usr/bin/env python3

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import trimesh
from pathlib import Path
from matplotlib import cm
from scipy.spatial import cKDTree
from plyfile import PlyData

# f3rm/CLIP imports (make sure they're installed in your environment)
from f3rm.features.clip import clip, tokenize
from f3rm.features.clip.model import CLIP
from f3rm.features.clip_extract import CLIPArgs


def get_heatmap(values, invert=False, cmap_name="plasma", normalize_range=(0.0, 1.0), only_show_match=True, threshold=0.5):
    """
    Return an RGBA colormap for the given values.
    """
    if len(values) == 0:
        return np.zeros((0, 4))

    values_np = values.cpu().numpy()  # shape (N,)
    cmap = cm.get_cmap(cmap_name)

    min_val, max_val = normalize_range
    val_min, val_max = values_np.min(), values_np.max()

    if val_max != val_min:
        normalized = (values_np - val_min) / (val_max - val_min)
        normalized = normalized * (max_val - min_val) + min_val
        normalized = np.clip(normalized, min_val, max_val)
        normalized_for_cmap = (normalized - min_val) / (max_val - min_val)
    else:
        normalized_for_cmap = np.zeros_like(values_np)

    if invert:
        normalized_for_cmap = 1 - normalized_for_cmap

    colors = cmap(normalized_for_cmap)

    if only_show_match:
        # Mask out points below threshold
        norm_threshold = (threshold - min_val) / (max_val - min_val) if max_val != min_val else 0
        mask = normalized_for_cmap < norm_threshold
        colors[mask, 3] = 0.0

    return colors


def get_initial_voxel_grid_from_saved(
    grid_feature_path: str,
    query: str,
    clip_model,
    device: str = "cuda",
    alpha_threshold: float = 0.01,
    softmax_temperature: float = 0.1,
):
    """
    Load voxel data (alphas & CLIP features) from disk, then filter:
      1) density threshold
      2) gray background threshold
      3) single-text-query filtering

    Returns:
      coords_filtered: shape (M, 3)
      voxel_sims_filtered: shape (M,)
      metrics: dict with counts
    """
    print(f"Loading feature grid from: {grid_feature_path}")
    metadata = np.load(grid_feature_path)
    min_bounds = metadata['min_bounds']
    max_bounds = metadata['max_bounds']
    grid_shape = metadata['grid_shape']

    print(f"Grid shape: {grid_shape}")
    print(f"Bounds: min={min_bounds}, max={max_bounds}")

    # Load corresponding .npy files
    base = grid_feature_path.replace('.npz', '')
    alphas = np.load(base + '_alphas.npy')    # shape (X, Y, Z)
    features = np.load(base + '_features.npy')# shape (X, Y, Z, feat_dim)
    rgb = np.load(base + '_rgb.npy')          # shape (X, Y, Z, 3)

    # Convert to torch
    alphas_tensor = torch.from_numpy(alphas).to(device)
    features_tensor = torch.from_numpy(features).to(device)
    rgb_tensor = torch.from_numpy(rgb).to(device)

    print("Creating coordinate grid...")
    x = torch.linspace(min_bounds[0], max_bounds[0], grid_shape[0], device=device)
    y = torch.linspace(min_bounds[1], max_bounds[1], grid_shape[1], device=device)
    z = torch.linspace(min_bounds[2], max_bounds[2], grid_shape[2], device=device)

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # shape (X, Y, Z, 3)

    coords_flat = coords.reshape(-1, 3)
    alphas_flat = alphas_tensor.reshape(-1)
    features_flat = features_tensor.reshape(-1, features_tensor.shape[-1])
    rgb_flat = rgb_tensor.reshape(-1, 3)

    metrics = {"initial_points": len(coords_flat)}

    # 1) Density threshold
    density_mask = alphas_flat > alpha_threshold
    coords_filtered = coords_flat[density_mask]
    features_filtered = features_flat[density_mask]
    rgb_filtered = rgb_flat[density_mask]
    metrics["after_density"] = len(coords_filtered)
    print(f"After density filtering: {len(coords_filtered)} points")

    # 2) Gray threshold
    rgb_std = rgb_filtered.std(dim=-1)
    non_gray_mask = rgb_std > 0.05
    coords_filtered = coords_filtered[non_gray_mask]
    features_filtered = features_filtered[non_gray_mask]
    metrics["after_gray"] = len(coords_filtered)
    print(f"After gray filtering: {len(coords_filtered)} points")

    # 3) Language filtering (SINGLE QUERY)
    print(f"Applying language filtering for query: {query}")
    with torch.no_grad():
        text_inputs = clip.tokenize([query]).to(device)
        query_embs = clip_model.encode_text(text_inputs).float()   # shape (1, feat_dim)
        query_embs /= query_embs.norm(dim=-1, keepdim=True)

    features_filtered = features_filtered.float()
    features_filtered /= features_filtered.norm(dim=-1, keepdim=True)

    # shape of voxel_sims is (N, 1)
    voxel_sims = features_filtered @ query_embs.T
    probs = voxel_sims / softmax_temperature
    probs = torch.nn.functional.softmax(probs, dim=-1)
    probs = torch.nan_to_num(probs, nan=1e-7)

    # Keep if the label is the query (index=0)
    labels = torch.multinomial(probs, num_samples=1)
    softmax_mask = (labels == 0).squeeze()

    coords_filtered = coords_filtered[softmax_mask]
    voxel_sims_filtered = voxel_sims.squeeze()[softmax_mask]  # shape (M,)
    metrics["after_language"] = len(coords_filtered)
    print(f"After language filtering: {len(coords_filtered)} points")

    return coords_filtered, voxel_sims_filtered, metrics


def visualize_filtered_voxels(voxel_grid, voxel_sims=None, title="Filtered Voxels"):
    """
    Visualize the filtered voxels with an optional heatmap coloring.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    points = voxel_grid.cpu().numpy() if torch.is_tensor(voxel_grid) else voxel_grid

    if voxel_sims is not None:
        if torch.is_tensor(voxel_sims):
            voxel_sims = voxel_sims.cpu()
        colors = get_heatmap(voxel_sims, only_show_match=False)
    else:
        colors = np.ones((len(points), 3)) * 0.5

    # colorbar
    cbar = plt.colorbar(cm.ScalarMappable(cmap=cm.plasma), ax=ax)
    cbar.set_label('Similarity')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=5)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    return fig


def main(query='leaves',
         result_dir="outputs/ecb91f433f144a7798724890f0528b23/f3rm/2025-03-24_235407",
         threshold=0.2):
    """
    1) Loads `pc.ply` from result_dir (dense point cloud).
    2) Loads `clip_features.npz` from result_dir (voxel data), runs CLIP-based filtering with `query`.
    3) Saves the filtered voxel data (xyz + sims) to ./outputs.
    4) Reads the same dense PLY again (or from memory) and uses a KD-tree
       to map each dense point to its nearest voxel's similarity. 
    5) Thresholds those sims and saves final segmentation.
    """
    device = "cuda"

    # ------------------- PART 1: VOXEL FILTERING -------------------
    clip_path = f"{result_dir}/clip_features.npz"
    if not os.path.exists(clip_path):
        raise FileNotFoundError(f"Could not find clip_features.npy at: {clip_path}")

    print(f"\n--- PART 1: CLIP-based voxel filtering with query='{query}' ---")
    # Load CLIP
    clip_model, _ = clip.load(CLIPArgs.model_name, device=device)

    # Filtered voxel grid
    voxel_grid, voxel_sims, metrics = get_initial_voxel_grid_from_saved(
        grid_feature_path=clip_path,
        query=query,
        clip_model=clip_model,
        device=device
    )

    # Visualize and save
    out_dir = Path("./outputs")
    out_dir.mkdir(exist_ok=True, parents=True)

    fig = visualize_filtered_voxels(voxel_grid, voxel_sims, title=f"Filtered Voxels: {query}")
    fig.savefig(out_dir / f"{query}_filtered_voxels.png", dpi=300)
    plt.close(fig)

    np.save(out_dir / f"{query}_filtered_voxel_grid.npy", voxel_grid.cpu().numpy() if torch.is_tensor(voxel_grid) else voxel_grid)
    np.save(out_dir / f"{query}_filtered_voxel_sims.npy", voxel_sims.cpu().numpy() if torch.is_tensor(voxel_sims) else voxel_sims)

    with open(out_dir / f"{query}_metrics.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    # ------------------- PART 2: KD-TREE DENSE MAPPING -------------------
    print(f"\n--- PART 2: KD-tree mapping for dense point cloud (threshold={threshold}) ---")
    ply_file = f"{result_dir}/pc.ply"
    if not os.path.exists(ply_file):
        raise FileNotFoundError(f"Could not find pc.ply at: {ply_file}")

    # Read dense data from .ply
    # (Use plyfile to match your original snippet exactly.)
    # Or we can rely on trimesh.load(ply_file), but let's do the user snippet approach.
    from plyfile import PlyData
    ply_data = PlyData.read(ply_file)
    vertex_data = ply_data['vertex'].data
    xyz = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T

    print("Original PLY shape:", xyz.shape)

    # Filter by Z <= 0.5 (example)
    filtered_xyz = xyz[xyz[:, 2] <= 0.5]
    print("Filtered PLY shape:", filtered_xyz.shape)

    # Load the new voxel data & sims
    voxel_data = np.load(out_dir / f"{query}_filtered_voxel_grid.npy")  # shape (M, 3)
    sims = np.load(out_dir / f"{query}_filtered_voxel_sims.npy")        # shape (M,)

    print("Voxel grid shape:", voxel_data.shape)
    print("Similarity shape:", sims.shape)

    # Threshold for being assigned to object
    thresholded_sims = (sims > threshold).astype(np.int8)

    # Build KD-tree on the voxel data
    kdtree = cKDTree(voxel_data)

    # Query nearest neighbor
    distances, nn_indices = kdtree.query(filtered_xyz, k=1)
    assigned_similarities = thresholded_sims[nn_indices]

    print(f"Min distance: {distances.min():.4f}, Max distance: {distances.max():.4f}")

    seg_dir = Path('./outputs_segmentation')
    seg_dir.mkdir(exist_ok=True, parents=True)
    seg_path = seg_dir / "dense_similarities.npy"
    np.save(seg_path, assigned_similarities)
    print(f"Saved assigned similarities to '{seg_path}'")

    # Visualize final dense segmentation
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        filtered_xyz[:, 0],
        filtered_xyz[:, 1],
        filtered_xyz[:, 2],
        s=1,
        c=assigned_similarities,
        cmap='viridis'
    )
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Thresholded Similarity (0 or 1)')

    all_points = np.concatenate([filtered_xyz, voxel_data], axis=0)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"KD-tree mapping (threshold={threshold})")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified script for CLIP-based voxel filtering + KD-tree mapping.")
    parser.add_argument("--query", type=str, default="leaves", 
                        help="Query string for CLIP-based filtering (default: 'leaves').")
    parser.add_argument("--result_dir", type=str, default="nerf_model/",
                        help="Path containing 'pc.ply' and 'clip_features.npz' (default: the sample path).")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Similarity threshold for final KD-tree assignment (default: 0.2).")
    args = parser.parse_args()

    main(query=args.query, result_dir=args.result_dir, threshold=args.threshold)
