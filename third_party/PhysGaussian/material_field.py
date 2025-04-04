import numpy as np
import torch
import json
import os
import warp as wp
from tqdm import tqdm
from sklearn.cluster import DBSCAN

def apply_material_field_to_simulation(mpm_solver, params, device="cuda:0"):
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
    
    assert len(part_labels) == n_particles, f"Number of particles ({n_particles}) doesn't match material field data ({len(part_labels)})"
    # Use material dictionary set_parameters_dict which already handles this functionality
    material_params = {
        "additional_material_params": []
    }
    
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

    # # For each particle, create a tiny region containing just that particle
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


        # # ## NOTE: use the boundary condition hack in the original PhysGaussian code
        # if material_ids[i] == 6:
        #     mpm_solver.set_velocity_on_cuboid(
        #         point=pos.tolist(),
        #         size=[0.001, 0.001, 0.001],
        #         velocity=[0.0, 0.0, 0.0],
        #         start_time=0.0,
        #         end_time=1e3,
        #         reset=1
        #     )
        
    
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

def apply_similarity_based_materials_to_simulation(mpm_solver, similarity_path, 
                                                  light_material, stiff_material,
                                                  device="cuda:0"):
    """
    Apply material properties to particles based on similarity values.
    """
    import warp as wp
    import numpy as np
    from mpm_solver_warp.mpm_utils import set_value_to_float_array, get_float_array_product
    
    # Load the similarity data
    similarities = np.load(similarity_path).flatten()
    
    # Make sure the number of particles matches
    n_particles = mpm_solver.n_particles
    if len(similarities) != n_particles:
        print(f"Warning: Number of particles ({n_particles}) doesn't match similarity data ({len(similarities)})")
        # Resize the array to match
        if len(similarities) > n_particles:
            similarities = similarities[:n_particles]
        else:
            similarities = np.pad(similarities, (0, n_particles - len(similarities)), 'constant')
    
    # Create arrays for each property with interpolated values
    E_values = np.zeros(n_particles, dtype=np.float32)
    nu_values = np.zeros(n_particles, dtype=np.float32)
    density_values = np.zeros(n_particles, dtype=np.float32)
    
    # Interpolate properties based on similarity values
    for i in range(n_particles):
        sim_value = float(similarities[i])
        # Linear interpolation between stiff and light material properties
        E_values[i] = stiff_material["E"] * (1 - sim_value) + light_material["E"] * sim_value
        nu_values[i] = stiff_material["nu"] * (1 - sim_value) + light_material["nu"] * sim_value
        density_values[i] = stiff_material["density"] * (1 - sim_value) + light_material["density"] * sim_value
    
    # Use material dictionary set_parameters_dict which already handles this functionality
    material_params = {
        "additional_material_params": []
    }
    
    # For each particle, create a tiny region containing just that particle
    for i in range(n_particles):
        if i % 10000 == 0:  # Print progress every 10000 particles
            print(f"Processing particle {i}/{n_particles}")
            
        # Get particle position
        pos = mpm_solver.mpm_state.particle_x.numpy()[i]
        
        # Add a material region for this particle
        material_params["additional_material_params"].append({
            "point": pos.tolist(),
            "size": [0.001, 0.001, 0.001],  # Tiny region containing just this particle
            "density": float(density_values[i]),
            "E": float(E_values[i]),
            "nu": float(nu_values[i])
        })
    
    # Apply these parameters
    mpm_solver.set_parameters_dict(material_params, device=device)
    
    # Finalize by computing mu and lambda parameters
    mpm_solver.finalize_mu_lam(device=device)
    
    # Print a summary of material property distribution
    stiff_count = np.sum(similarities < 0.25)
    medium_count = np.sum((similarities >= 0.25) & (similarities < 0.75))
    light_count = np.sum(similarities >= 0.75)
    
    print("\n==== Material Property Distribution Summary ====")
    print(f"Total particles: {n_particles}")
    print(f"Stiff particles (similarity < 0.25): {stiff_count} ({stiff_count/n_particles*100:.1f}%)")
    print(f"  - Young's modulus: ~{stiff_material['E']:.1e}")
    print(f"  - Poisson's ratio: ~{stiff_material['nu']:.2f}")
    print(f"  - Density: ~{stiff_material['density']:.1f}")
    
    print(f"Medium particles (0.25 ≤ similarity < 0.75): {medium_count} ({medium_count/n_particles*100:.1f}%)")
    middle_E = (stiff_material["E"] + light_material["E"]) / 2
    middle_nu = (stiff_material["nu"] + light_material["nu"]) / 2
    middle_density = (stiff_material["density"] + light_material["density"]) / 2
    print(f"  - Young's modulus: ~{middle_E:.1e}")
    print(f"  - Poisson's ratio: ~{middle_nu:.2f}")
    print(f"  - Density: ~{middle_density:.1f}")
    
    print(f"Light particles (similarity ≥ 0.75): {light_count} ({light_count/n_particles*100:.1f}%)")
    print(f"  - Young's modulus: ~{light_material['E']:.1e}")
    print(f"  - Poisson's ratio: ~{light_material['nu']:.2f}")
    print(f"  - Density: ~{light_material['density']:.1f}")
    print("================================================\n")
    
    # Also print the min, max, and mean values for each property
    print("Actual property ranges:")
    print(f"Young's modulus: min={np.min(E_values):.1e}, max={np.max(E_values):.1e}, mean={np.mean(E_values):.1e}")
    print(f"Poisson's ratio: min={np.min(nu_values):.2f}, max={np.max(nu_values):.2f}, mean={np.mean(nu_values):.2f}")
    print(f"Density: min={np.min(density_values):.1f}, max={np.max(density_values):.1f}, mean={np.mean(density_values):.1f}")
    
    print(f"Applied similarity-based material properties to {n_particles} particles")
    return similarities

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