import torch
import numpy as np
from typing import Tuple, Dict, List
import trimesh
from f3rm.features.clip import clip
from f3rm.features.clip_extract import CLIPArgs
from utils import str2bool
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
        m = mode(neighbor_labels, keepdims=False)
        new_labels_np[i] = m.mode
    
    # Return as a torch tensor on the original device
    return torch.tensor(new_labels_np, device=part_labels.device)



def save_segmented_point_cloud(
    coords: torch.Tensor,
    part_labels: torch.Tensor,
    output_path: str,
    part_scores: torch.Tensor = None,
    use_scores_for_alpha: bool = False,
    cmap_name: str = 'tab10',
    original_pc_path: str = None,
    part_queries: List[str] = None,
    material_dict_path: str = None
):
    """
    Save segmented point cloud to a PLY file with colors based on part labels.
    
    Args:
        coords: Tensor of shape (num_points, 3) containing point coordinates
        part_labels: Tensor of shape (num_points) containing part indices
        output_path: Path to save the PLY file
        part_scores: Optional tensor of shape (num_points) with similarity scores
        use_scores_for_alpha: If True, use scores to determine point alpha
        cmap_name: Name of the colormap to use for part colors
        original_pc_path: Path to the original point cloud file (required if use_actual_rgb=True)
        part_queries: List of part query strings corresponding to part_labels
        material_dict_path: Path to JSON file mapping part queries to material properties
    """
    import numpy as np
    from plyfile import PlyData, PlyElement
    import matplotlib.pyplot as plt
    import trimesh
    import json
    import os
    
    # Convert tensors to numpy arrays
    coords_np = coords.cpu().numpy()
    part_labels_np = part_labels.cpu().numpy()
    
    # Initialize colors array for RGB and semantic colors
    rgb_colors = np.zeros((coords_np.shape[0], 4), dtype=np.float32)
    semantic_colors = np.zeros((coords_np.shape[0], 4), dtype=np.float32)
    
    # Initialize material property arrays
    density = np.zeros(coords_np.shape[0], dtype=np.float32)
    E = np.zeros(coords_np.shape[0], dtype=np.float32)
    nu = np.zeros(coords_np.shape[0], dtype=np.float32)
    material_id = np.zeros(coords_np.shape[0], dtype=np.int32)
    
    # Load material properties dictionary if provided
    material_props = {}
    if material_dict_path and part_queries:
        try:
            with open(material_dict_path, 'r') as f:
                material_props = json.load(f)
            print(f"Loaded material properties from {material_dict_path}")
        except Exception as e:
            print(f"Error loading material properties: {e}")
            print("Using default material properties instead")
    
    # Get RGB colors from original point cloud if available
    if original_pc_path:
        print(">>> LOADING ORIGINAL RGB")
        # Load original point cloud to get RGB values
        original_pc = trimesh.load(original_pc_path)
        original_vertices = np.asarray(original_pc.vertices)
        original_colors = np.asarray(original_pc.colors)
        
        # Normalize colors if needed
        if original_colors.max() > 1.0:
            original_colors = original_colors / 255.0
            
        # We need to map the filtered coordinates back to the original point cloud
        # This is a simple implementation that finds the nearest neighbor
        from scipy.spatial import cKDTree
        tree = cKDTree(original_vertices)
        _, indices = tree.query(coords_np, k=1)
        
        # Get the corresponding colors
        rgb_colors[:, :3] = original_colors[indices, :3]
        rgb_colors[:, 3] = 1.0  # Full alpha
    else:
        # If no original point cloud, use white for RGB
        rgb_colors[:, :3] = 1.0  # White
        rgb_colors[:, 3] = 1.0  # Full alpha
    
    # Create semantic colors based on part labels
    print(">>> CREATING SEMANTIC COLORS")
    # Create a colormap with distinct colors for each part
    num_parts = part_labels_np.max() + 1
    cmap = plt.colormaps[cmap_name]
    
    # Generate colors for each point based on its part label
    for i in range(num_parts):
        mask = (part_labels_np == i)
        if not np.any(mask):
            continue
            
        base_color = cmap(i % cmap.N)  # RGBA tuple
        
        if use_scores_for_alpha and part_scores is not None:
            # Use scores for alpha
            alphas = part_scores.cpu().numpy()[mask]
            alphas = np.clip(alphas, 0.1, 1.0)  # Ensure minimum visibility
            
            # Set RGB values
            semantic_colors[mask, :3] = np.array(base_color[:3])
            # Set alpha values
            semantic_colors[mask, 3] = alphas
        else:
            # Use fixed alpha
            semantic_colors[mask] = np.array(base_color)
    
    # Assign material properties based on part labels
    for i in range(part_labels_np.max() + 1):
        mask = (part_labels_np == i)
        if not np.any(mask):
            continue
        
        # Get part query string for this label
        part_name = part_queries[i] if part_queries and i < len(part_queries) else f"part_{i}"
        
        # Get material properties for this part
        if part_name in material_props:
            props = material_props[part_name]
            density[mask] = props.get("density", 200)
            E[mask] = props.get("E", 2e6)
            nu[mask] = props.get("nu", 0.4)
            material_id[mask] = props.get("material_id", 0)
            print(f"Applied material properties for {part_name}: {props}")
        else:
            # Default values if part not found in material dictionary
            density[mask] = 200
            E[mask] = 2e6
            nu[mask] = 0.4
            material_id[mask] = 0
            print(f"Using default material properties for {part_name}")
    
    # Save both RGB and semantic point clouds
    
    # 1. Save RGB point cloud
    # Convert floating point colors [0,1] to uint8 [0,255]
    rgb_colors_uint8 = (rgb_colors * 255).astype(np.uint8)
    
    # Create structured array for RGB PLY file
    vertex_data_rgb = np.zeros(
        coords_np.shape[0],
        dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1'),
            ('part_label', 'i4'), ('density', 'f4'), ('E', 'f4'), ('nu', 'f4'),
            ('material_id', 'i4')
        ]
    )
    
    # Fill in the data for RGB point cloud
    vertex_data_rgb['x'] = coords_np[:, 0]
    vertex_data_rgb['y'] = coords_np[:, 1]
    vertex_data_rgb['z'] = coords_np[:, 2]
    vertex_data_rgb['red'] = rgb_colors_uint8[:, 0]
    vertex_data_rgb['green'] = rgb_colors_uint8[:, 1]
    vertex_data_rgb['blue'] = rgb_colors_uint8[:, 2]
    vertex_data_rgb['alpha'] = rgb_colors_uint8[:, 3]
    vertex_data_rgb['part_label'] = part_labels_np
    vertex_data_rgb['density'] = density
    vertex_data_rgb['E'] = E
    vertex_data_rgb['nu'] = nu
    vertex_data_rgb['material_id'] = material_id
    
    # Create PLY element and save RGB file
    vertex_element_rgb = PlyElement.describe(vertex_data_rgb, 'vertex')
    PlyData([vertex_element_rgb], text=False).write(output_path)
    print(f"RGB point cloud saved to {output_path}")
    
    # 2. Save semantic point cloud
    # Generate semantic output path
    base_name, ext = os.path.splitext(output_path)
    semantic_output_path = f"{base_name}_semantics{ext}"
    
    # Convert semantic colors to uint8
    semantic_colors_uint8 = (semantic_colors * 255).astype(np.uint8)
    
    # Create structured array for semantic PLY file
    vertex_data_semantic = np.zeros(
        coords_np.shape[0],
        dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1'),
            ('part_label', 'i4'), ('density', 'f4'), ('E', 'f4'), ('nu', 'f4'),
            ('material_id', 'i4')
        ]
    )
    
    # Fill in the data for semantic point cloud
    vertex_data_semantic['x'] = coords_np[:, 0]
    vertex_data_semantic['y'] = coords_np[:, 1]
    vertex_data_semantic['z'] = coords_np[:, 2]
    vertex_data_semantic['red'] = semantic_colors_uint8[:, 0]
    vertex_data_semantic['green'] = semantic_colors_uint8[:, 1]
    vertex_data_semantic['blue'] = semantic_colors_uint8[:, 2]
    vertex_data_semantic['alpha'] = semantic_colors_uint8[:, 3]
    vertex_data_semantic['part_label'] = part_labels_np
    vertex_data_semantic['density'] = density
    vertex_data_semantic['E'] = E
    vertex_data_semantic['nu'] = nu
    vertex_data_semantic['material_id'] = material_id
    
    # Create PLY element and save semantic file
    vertex_element_semantic = PlyElement.describe(vertex_data_semantic, 'vertex')
    PlyData([vertex_element_semantic], text=False).write(semantic_output_path)
    print(f"Semantic point cloud saved to {semantic_output_path}")



import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_feature_path", type=str, required=True)
    parser.add_argument("--occupancy_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--part_queries", type=str, required=True)
    parser.add_argument("--material_dict_path", type=str, default=None, 
                        help="Path to JSON file mapping part queries to material properties")
    parser.add_argument("--use_spatial_smoothing", type=str2bool, default=False)
    args = parser.parse_args()

    # Parse part queries from comma-separated string
    part_queries = [q.strip() for q in args.part_queries.split(',')]
    print(">> PART_QUERIES", part_queries)
    
    coords_filtered, part_labels, part_scores, metrics = clip_part_segmentation(args.grid_feature_path, 
                                                                                part_queries,
                                                                                args.occupancy_path)
    if args.use_spatial_smoothing:
        part_labels = local_post_process_segmentation(coords_filtered, part_labels)
    
    # Always save both RGB and semantic point clouds
    save_segmented_point_cloud(coords_filtered, part_labels, args.output_path, part_scores,
                               original_pc_path=args.occupancy_path,
                               part_queries=part_queries, material_dict_path=args.material_dict_path)