import numpy as np
import torch
import json
import os
import warp as wp
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from utils.transformation_utils import *



def visualize_positions(material_positions, mpm_positions, output_path="material_field_debug"):
    """
    Visualize material positions and MPM positions in 3D.
    
    Args:
        material_positions: Numpy array of material field positions
        mpm_positions: Numpy array of MPM solver positions
        output_path: Directory to save the visualization
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot material positions in red
    ax.scatter(material_positions[:, 0], material_positions[:, 1], material_positions[:, 2], 
               c='red', s=2, alpha=0.5, label='Material Field Points')
    
    # Plot MPM positions in blue
    ax.scatter(mpm_positions[:, 0], mpm_positions[:, 1], mpm_positions[:, 2], 
               c='blue', s=2, alpha=0.5, label='MPM Solver Points')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Material Field vs MPM Solver Positions')
    ax.legend()
    
    # Save figure
    plt.savefig(os.path.join(output_path, "position_comparison.png"), dpi=150, bbox_inches='tight')
    
    # Create a second figure showing a 2D projection (top view)
    fig2 = plt.figure(figsize=(12, 10))
    ax2 = fig2.add_subplot(111)
    
    # Plot material positions in red
    ax2.scatter(material_positions[:, 0], material_positions[:, 1], 
                c='red', s=2, alpha=0.5, label='Material Field Points')
    
    # Plot MPM positions in blue
    ax2.scatter(mpm_positions[:, 0], mpm_positions[:, 1], 
                c='blue', s=2, alpha=0.5, label='MPM Solver Points')
    
    # Set labels and title
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Material Field vs MPM Solver Positions (Top View)')
    ax2.legend()
    
    # Save figure
    plt.savefig(os.path.join(output_path, "position_comparison_top.png"), dpi=150, bbox_inches='tight')
    
    plt.close('all')
    print(f"Visualization saved to {output_path}")


def apply_material_field_to_simulation(mpm_solver, params, device="cuda:0",
                                       scale_origin=None, original_mean_pos=None, rotation_matrices=None):
    """
    Apply material properties to particles based on material field data loaded from a point cloud.
    
    Args:
        mpm_solver: The MPM solver instance
        params: Dictionary containing material properties extracted from the point cloud
        device: Device to run computations on
    """
    # Check if material properties exist in the params
    assert all(key in params for key in ['part_labels', 'density', 'E', 'nu', 'material_id']), f"Missing material properties in params: {params.keys()}"
    
    # Get the number of particles
    n_particles = mpm_solver.n_particles
    
    # Extract material properties from params
    part_labels = params['part_labels'].cpu().numpy() if torch.is_tensor(params['part_labels']) else params['part_labels']
    densities = params['density'].cpu().numpy() if torch.is_tensor(params['density']) else params['density']
    E_values = params['E'].cpu().numpy() if torch.is_tensor(params['E']) else params['E']
    nu_values = params['nu'].cpu().numpy() if torch.is_tensor(params['nu']) else params['nu']
    material_ids = params['material_id'].cpu().numpy() if torch.is_tensor(params['material_id']) else params['material_id']
    
    # If the number of particles doesn't match, perform nearest neighbor interpolation
    if len(part_labels) != n_particles:
        print(f"Material field data ({len(part_labels)} particles) doesn't match MPM solver ({n_particles} particles). Performing nearest neighbor interpolation.")
        
        # Get positions from both the material field and the MPM solver
        material_positions = params['pos'].cpu().numpy() if torch.is_tensor(params['pos']) else params['pos']
        mpm_positions = mpm_solver.export_particle_x_to_torch().to(device)
        mpm_positions = apply_inverse_rotations(
                undotransform2origin(
                    undoshift2center111(mpm_positions), scale_origin, original_mean_pos
                ),
                rotation_matrices,
            ).detach().cpu().numpy()

        visualize_positions(material_positions, mpm_positions, output_path="material_field_debug")
        
        # Build the nearest neighbors model with the material field positions
        nn_model = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(material_positions)
        
        # Find the nearest neighbor for each MPM particle
        distances, indices = nn_model.kneighbors(mpm_positions)
        
        # Map material properties to MPM particles using the nearest neighbor indices
        part_labels = part_labels[indices.flatten()]
        densities = densities[indices.flatten()]
        E_values = E_values[indices.flatten()]
        nu_values = nu_values[indices.flatten()]
        material_ids = material_ids[indices.flatten()]
        
        print(f"Nearest neighbor interpolation complete. Average distance to nearest material point: {distances.mean():.6f}")
        # assert distances.mean() < 0.005, f"Average distance to nearest material point {distances.mean()} is too large"
    else:
        print(f"Material field data matches MPM solver ({n_particles} particles).")
    
    positions = mpm_solver.mpm_state.particle_x.numpy()
    handle_stationary_clusters(
        mpm_solver, 
        positions=positions, 
        material_ids=material_ids,
        eps=0.03, 
        min_samples=8, 
        start_time=0.0, 
        end_time=1e9,
        buffer=0.1,
    )

    # Create material parameters dictionary
    material_params = {
        "additional_material_params": []
    }

    # For each particle, create a tiny region containing just that particle
    for i in tqdm(range(n_particles), desc="Applying material field to particles"):
        # Get particle position
        pos = mpm_solver.mpm_state.particle_x.numpy()[i]
        # Add a material region for this particle
        material_params["additional_material_params"].append({
            "point": pos.tolist(),
            "size": [0.001, 0.001, 0.001],  # Tiny region containing just this particle
            "density": float(densities[i]),
            "E": float(E_values[i]),
            "nu": float(nu_values[i]),
            "material": int(material_ids[i]),
        })
    
    # Apply these parameters
    mpm_solver.set_parameters_dict(material_params, device=device)
    # Finalize by computing mu and lambda parameters
    mpm_solver.finalize_mu_lam(device=device)
    

def get_material_name(material_id):
    """
    Convert material ID to a human-readable name based on the MPM material types.
    
    Args:
        material_id: Integer material ID
        
    Returns:
        String name of the material
    """
    material_names = {
        0: "jelly",
        1: "metal",
        2: "sand",
        3: "visplas",
        4: "fluid",
        5: "snow",
        6: "stationary"
    }
    
    return material_names.get(material_id, "unknown")

def handle_stationary_clusters(mpm_solver, positions, material_ids,
                               eps=0.03, min_samples=10,
                               start_time=0.0, end_time=1e6, buffer=0.0):
    """
    Automatically clusters stationary particles and creates one cuboid BC per cluster.
    
    Args:
        mpm_solver: your MPM_Simulator_WARP or similar solver instance
        positions: (N, 3) numpy array of particle positions
        material_ids: length-N array of material IDs for each particle
        eps: DBSCAN max distance for two samples to be in the same neighborhood
        min_samples: DBSCAN min number of samples to form a dense region
        start_time: BC start time
        end_time: BC end time
        buffer: an optional buffer to extend each bounding box in all directions (in world units)
    """
    # 1) Filter only the stationary (material=6) particles
    stationary_mask = (material_ids == 6)
    print(">>> stationary_mask: ", stationary_mask, "Number of stationary particles: ", np.sum(stationary_mask),
          "material_ids: ", np.unique(material_ids))
    stationary_positions = positions[stationary_mask]
    if len(stationary_positions) == 0:
        print("No stationary particles found; skipping cluster-based cuboid BC creation.")
        return

    # 2) Run DBSCAN to find clusters among stationary positions
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(stationary_positions)

    unique_labels = np.unique(labels)
    if len(unique_labels) == 1 and unique_labels[0] == -1:
        print("All stationary points marked as noise by DBSCAN; no cuboid BCs created.")
        return

    # 3) For each cluster, compute bounding box and add one cuboid BC
    for cluster_id in unique_labels:
        if cluster_id == -1:  # DBSCAN noise
            continue

        cluster_points = stationary_positions[labels == cluster_id]
        min_xyz = cluster_points.min(axis=0)
        max_xyz = cluster_points.max(axis=0)

        print(">> MIN_XYZ: ", min_xyz, "max_xyz: ", max_xyz)

        # bounding box center
        center = 0.5 * (min_xyz + max_xyz)
        # half-size
        halfsize = 0.5 * (max_xyz - min_xyz)

        # add optional buffer
        halfsize += buffer
        # 4) Create a single velocity-on-cuboid boundary condition for this cluster
        #    velocity=0, effectively pins that region for the entire simulation
        mpm_solver.set_velocity_on_cuboid(
            point=center.tolist(),
            size=halfsize.tolist(),
            velocity=[0.0, 0.0, 0.0],
            start_time=start_time,
            end_time=end_time,
            reset=1   # reset=1 forcibly sets velocity each step
        )
        print(">>> Created cuboid BC for cluster ", cluster_id, " at ", center.tolist(), " with size ", halfsize.tolist())

    print(f"Created cuboid BC for {len(unique_labels[unique_labels != -1])} stationary cluster(s).")