import torch
import numpy as np
from typing import Tuple, Dict, List
import trimesh
from f3rm.features.clip import clip
from f3rm.features.clip_extract import CLIPArgs

def get_initial_voxel_grid_from_saved(
    grid_feature_path: str,
    occupancy_path: str = None,  # Add parameter for point cloud path
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    """
    Get the feature grid from a saved file masked by the occupancy grid provided
    by the occupancy_path. `occupancy_path` was precomputed using alpha thresholding +
    removing `black` background + connected component outlier removal using DBSCAN. See 
    `voxel_to_pc.py` for more details.
    
    Args:
        grid_feature_path: Path to the saved feature grid metadata (.npz file)
        query: Text query to filter voxels
        clip_model: CLIP model for text encoding
        device: Device to use for computation
        alpha_threshold: Threshold for density values
        softmax_temperature: Temperature for softmax when computing language probabilities
        voxel_size: Size of each voxel
        point_cloud_path: Path to pre-filtered point cloud (if None, will use default path)
        
    Returns:
        Tuple containing:
        - voxel_grid: Tensor of shape (num_voxels, 3) containing filtered voxel coordinates
        - voxel_sims: Tensor of shape (num_voxels) containing similarities with language query
        - metrics: Dictionary with metrics about filtering process
    """
    print(f"Loading feature grid from {grid_feature_path}...")
    
    # Load metadata
    metadata = np.load(grid_feature_path)
    min_bounds = metadata['min_bounds']
    max_bounds = metadata['max_bounds']
    grid_shape = metadata['grid_shape']
    
    print(f"Bounds: min={min_bounds}, max={max_bounds}")
    
    # Load features
    features_path = grid_feature_path.replace('.npz', '_features.npy')
    print(f"Loading features from {features_path}...")
    features = np.load(features_path)
    
    # Track metrics
    metrics = {"initial": np.prod(grid_shape)}
    
    # Load the pre-filtered point cloud
    if occupancy_path is None:
        occupancy_path = grid_feature_path.replace('.npz', '_pc.ply')
    
    print(f"Loading pre-filtered point cloud from {occupancy_path}...")
    pc = trimesh.load(occupancy_path)
    points = np.asarray(pc.vertices)
    
    # Convert points to tensor and move to device
    coords_filtered = torch.tensor(points, dtype=torch.float32, device=device)
    metrics["point_cloud"] = len(coords_filtered)
    print(f"Loaded {len(coords_filtered)} points from point cloud")
    
    # Create a grid of all possible coordinates
    print("Creating coordinate grid...")
    x = torch.linspace(min_bounds[0], max_bounds[0], grid_shape[0], device=device)
    y = torch.linspace(min_bounds[1], max_bounds[1], grid_shape[1], device=device)
    z = torch.linspace(min_bounds[2], max_bounds[2], grid_shape[2], device=device)
    
    # Find the nearest grid points for each point in the point cloud
    # This maps the point cloud points back to indices in the feature grid
    print("Finding nearest grid points for point cloud points...")
    
    # For each point in the point cloud, find the closest grid point
    # We'll do this by finding the closest index in each dimension
    indices_x = torch.abs(coords_filtered[:, 0].unsqueeze(1) - x.unsqueeze(0)).argmin(dim=1)
    indices_y = torch.abs(coords_filtered[:, 1].unsqueeze(1) - y.unsqueeze(0)).argmin(dim=1)
    indices_z = torch.abs(coords_filtered[:, 2].unsqueeze(1) - z.unsqueeze(0)).argmin(dim=1)
    
    # Convert to linear indices
    linear_indices = (indices_x * grid_shape[1] * grid_shape[2] + 
                      indices_y * grid_shape[2] + 
                      indices_z)
    
    # Get features for these indices
    features_tensor = torch.from_numpy(features).to(device)
    features_flat = features_tensor.reshape(-1, features_tensor.shape[-1])
    features_filtered = features_flat[linear_indices]
    return features_filtered, coords_filtered, metrics




def clip_part_segmentation(
   grid_feature_path: str,
   part_queries: List[str],
   occupancy_path: str = None,
   device: str = "cuda",
softmax_temperature: float = 0.1,  # Added temperature parameter for sharpening

):
    """
    Perform part-based segmentation of a voxel grid using CLIP.
    
    Assign each voxel to one of the part queries.
    
    Args:
        grid_feature_path: Path to the saved feature grid metadata (.npz file)
        part_queries: List of text queries representing different parts
        device: Device to use for computation
        occupancy_path: Path to pre-filtered point cloud
        
    Returns:
        Tuple containing:
        - coords_filtered: Tensor of shape (num_voxels, 3) containing voxel coordinates
        - part_labels: Tensor of shape (num_voxels) containing part indices (0 to len(part_queries)-1)
        - part_scores: Tensor of shape (num_voxels) containing similarity scores for the assigned parts
        - metrics: Dictionary with metrics about the segmentation process
    """
    features_filtered, coords_filtered, metrics = get_initial_voxel_grid_from_saved(
        grid_feature_path,
        device=device,
        occupancy_path=occupancy_path,
    )
    
    # Load CLIP model
    clip_model, _ = clip.load(CLIPArgs.model_name, device=device)
    
    # Normalize features
    features_filtered = features_filtered.to(torch.float32)
    features_filtered /= features_filtered.norm(dim=-1, keepdim=True)
    
    # Encode all part queries
    with torch.no_grad():
        text_inputs = clip.tokenize(part_queries).to(device)
        query_embs = clip_model.encode_text(text_inputs).float()
        query_embs /= query_embs.norm(dim=-1, keepdim=True)
    
    # Compute similarities between each voxel and each part query
    # Shape: (num_voxels, num_parts)
    similarities = features_filtered @ query_embs.T


    scaled_similarities = similarities / softmax_temperature
    
    # Convert scaled similarities to probabilities via softmax.
    probabilities = torch.nn.functional.softmax(scaled_similarities, dim=1)


    
    # Get the index of the part with highest similarity for each voxel
    # Shape: (num_voxels)
    part_labels = torch.argmax(probabilities, dim=1)
    
    # Get the similarity score for the assigned part
    # Shape: (num_voxels)
    part_scores = torch.gather(probabilities, 1, part_labels.unsqueeze(1)).squeeze(1)
    
    
    # Get the coordinates for each voxel (assuming they're available from the first function)
    # This needs to be fixed as coords_filtered isn't returned by get_initial_voxel_grid_from_saved
    # For now, we'll need to reconstruct the coordinates
    
    metrics["num_parts"] = len(part_queries)
    
    # Count voxels assigned to each part
    for i, query in enumerate(part_queries):
        part_count = (part_labels == i).sum().item()
        metrics[f"part_{i}_{query}"] = part_count
        print(f"Part {i} ({query}): {part_count} voxels")
    
    return coords_filtered, part_labels, part_scores, metrics


def local_post_process_segmentation(
    coords: torch.Tensor,
    part_labels: torch.Tensor,
    k: int = 200,
) -> torch.Tensor:
    """
    Perform local post-processing on segmentation results using k-nearest neighbors majority voting.
    
    Args:
        coords: Tensor of shape (num_points, 3) containing point coordinates.
        part_labels: Tensor of shape (num_points) containing segmentation labels.
        k: Number of nearest neighbors to consider for voting.
        
    Returns:
        new_labels: Tensor of shape (num_points) with updated labels after local post-processing.
    """
    import numpy as np
    from sklearn.neighbors import KDTree
    from scipy.stats import mode

    # Convert tensors to NumPy arrays
    coords_np = coords.cpu().numpy()
    labels_np = part_labels.cpu().numpy()
    
    # Build a KDTree for fast neighbor search
    tree = KDTree(coords_np)
    new_labels_np = labels_np.copy()
    
    # For each point, query the k nearest neighbors and take a majority vote.
    for i, point in enumerate(coords_np):
        # Query the k nearest neighbors (including the point itself)
        _, indices = tree.query(point.reshape(1, -1), k=k)
        neighbor_labels = labels_np[indices[0]]
        # Compute the mode (most frequent label) among the neighbors
        m = mode(neighbor_labels)
        new_labels_np[i] = m.mode[0]
    
    # Return as a torch tensor on the original device
    return torch.tensor(new_labels_np, device=part_labels.device)
