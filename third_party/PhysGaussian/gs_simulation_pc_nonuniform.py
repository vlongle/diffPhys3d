import sys

sys.path.append("gaussian-splatting")

import argparse
import math
import cv2
import torch
import os
import numpy as np
import json
from tqdm import tqdm
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from third_party.PhysGaussian.material_field import apply_similarity_based_materials_to_simulation

# Gaussian splatting dependencies
from utils.sh_utils import eval_sh
from scene.gaussian_model import GaussianModel
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.cameras import Camera as GSCamera
from gaussian_renderer import render, GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import focal2fov

# MPM dependencies
from mpm_solver_warp.engine_utils import *
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
import warp as wp

# Particle filling dependencies
from particle_filling.filling import *

# Utils
from utils.decode_param import *
from utils.transformation_utils import *
from utils.camera_view_utils import *
from utils.render_utils import *
import glob    
from plyfile import PlyData
import numpy as np

wp.init()
wp.config.verify_cuda = True

ti.init(arch=ti.cuda, device_memory_GB=8.0)


def load_point_cloud(ply_path, opacity_value=0.5, sh_degree=3):
    """
    Load a point cloud from a PLY file and prepare it for simulation.
    
    Args:
        ply_path: Path to the PLY file
        opacity_value: Default opacity to assign to all points
        sh_degree: SH degree for the Gaussian model
    
    Returns:
        Dictionary with positions, default covariances, and opacities
    """

    
    # Load the point cloud
    plydata = PlyData.read(ply_path)
    vertex_element = plydata['vertex']
    
    # Print the PLY structure for debugging
    print("plydata", plydata)
    
    # Extract positions
    x = vertex_element['x']
    y = vertex_element['y']
    z = vertex_element['z']
    positions = np.column_stack((x, y, z))
    
    # Check if color data exists in the PLY file
    # Get property names from the vertex element
    property_names = [p.name for p in vertex_element.properties]
    has_colors = all(color in property_names for color in ['red', 'green', 'blue'])
    
    colors_tensor=None
    if has_colors:
        # Extract colors if they exist
        red = vertex_element['red']
        green = vertex_element['green']
        blue = vertex_element['blue']
        colors = np.column_stack((red, green, blue))
        colors_tensor = torch.tensor(colors, dtype=torch.float32, device="cuda") / 255.0
        
        # Convert colors to DC component (first spherical harmonic)
        features_dc = colors_tensor.unsqueeze(1) / 0.282095  # SH normalization factor
    else:
        # Default to gray color if no colors are present
        features_dc = torch.ones((positions.shape[0], 1, 3), device="cuda") * 0.5
    
    # Create rest of spherical harmonics
    features_rest = torch.zeros((positions.shape[0], (sh_degree + 1) ** 2 - 1, 3), device="cuda")
    
    # Combine into final SH coefficients
    shs = torch.cat([features_dc, features_rest], dim=1)
    
    # Create default covariances (small spheres)
    default_scale = 0.01  # Small default scale
    cov = torch.ones((positions.shape[0], 6), device="cuda") * default_scale
    
    # Create default opacities
    opacities = torch.ones((positions.shape[0], 1), device="cuda") * opacity_value
    
    # Create placeholder for screen points
    screen_points = torch.zeros((positions.shape[0], 3), device="cuda")
    
    positions_tensor = torch.tensor(positions, device="cuda").float()
    
    print(f"Loaded point cloud with {positions.shape[0]} points")
    if has_colors:
        print("Color information was successfully loaded")
    else:
        print("No color information found, using default gray color")
    
    return {
        "pos": positions_tensor,
        "cov3D_precomp": cov,
        "opacity": opacities,
        "shs": shs,
        "screen_points": screen_points,
        "colors": colors_tensor
    }


def save_point_cloud_image(positions, output_path, frame_number, color=None, 
                          padding=0.1, elevation=30, azimuth=45, dpi=200, point_size=1):
    """
    Save a simple visualization of a point cloud as an image using matplotlib,
    with automatic adjustment of view bounds to ensure the whole object is visible.
    
    Args:
        positions: Tensor of 3D point positions [N, 3]
        output_path: Directory to save the rendered image
        frame_number: Current frame number for filename
        color: Optional color array for points
        padding: Padding factor around the data bounds (0.1 = 10% extra space)
        elevation: Camera elevation angle
        azimuth: Camera azimuth angle
        dpi: Resolution of the output image
        point_size: Size of points in the plot
    """
    # Convert positions to numpy
    positions_np = positions.cpu().numpy()
    
    # Create figure
    fig = plt.figure(figsize=(10, 10), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    # If no color is provided, color by depth (Z coordinate)
    if color is None:
        color = positions_np[:, 2]
        
    # Plot the point cloud
    scatter = ax.scatter(
        positions_np[:, 0], 
        positions_np[:, 1], 
        positions_np[:, 2], 
        c=color, 
        s=point_size,  # Point size
        cmap='viridis',  # You can change the colormap as needed
        alpha=0.8
    )
    
    # Set the view angle
    ax.view_init(elev=elevation, azim=azimuth)
    
    # Calculate bounds with padding
    min_x, max_x = positions_np[:, 0].min(), positions_np[:, 0].max()
    min_y, max_y = positions_np[:, 1].min(), positions_np[:, 1].max()
    min_z, max_z = positions_np[:, 2].min(), positions_np[:, 2].max()
    
    # Calculate range of each dimension
    x_range = max_x - min_x
    y_range = max_y - min_y
    z_range = max_z - min_z
    
    # Find the largest range to maintain aspect ratio
    max_range = max(x_range, y_range, z_range)
    
    # Add padding
    padding_amount = max_range * padding
    
    # Set limits based on data with padding
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2
    mid_z = (min_z + max_z) / 2
    
    # Set equal aspect ratio limits centered on the object's midpoint
    ax.set_xlim(mid_x - max_range / 2 - padding_amount, mid_x + max_range / 2 + padding_amount)
    ax.set_ylim(mid_y - max_range / 2 - padding_amount, mid_y + max_range / 2 + padding_amount)
    ax.set_zlim(mid_z - max_range / 2 - padding_amount, mid_z + max_range / 2 + padding_amount)
    
    # Make sure all axes have equal scale
    ax.set_box_aspect([1, 1, 1])
    
    # Remove axes for cleaner visualization
    ax.set_axis_off()
    
    # Tight layout to maximize figure size
    plt.tight_layout()
    
    # Save the image
    output_file = os.path.join(output_path, f"{frame_number:04d}.png")
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    print(f"Saved frame {frame_number} to {output_file}")
    
    # Return dimensions of the saved image
    return dpi * 10, dpi * 10  # Approximate size based on figsize and dpi


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_ply", action="store_true")
    parser.add_argument("--output_h5", action="store_true")
    parser.add_argument("--render_img", action="store_true")
    parser.add_argument("--compile_video", action="store_true")
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--point_cloud_path", type=str, help="Path to input point cloud PLY file")
    parser.add_argument("--similarity_path", type=str, default="../../segmentation/outputs/dense_similarities.npy", 
                        help="Path to similarity data (.npy file)")
    args = parser.parse_args()

        
    if not os.path.exists(args.config):
        raise AssertionError("Scene config does not exist!")
    if args.output_path is not None and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # load scene config
    print("Loading scene config...")
    (
        material_params,
        bc_params,
        time_params,
        preprocessing_params,
        camera_params,
    ) = decode_param_json(args.config)

    print(f"Loading point cloud from {args.point_cloud_path}...")
    if not os.path.exists(args.point_cloud_path):
        raise AssertionError("Point cloud path does not exist!")
    params = load_point_cloud(args.point_cloud_path)

    # Set background color
    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        if args.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    )

    # init the scene
    print("Initializing scene and pre-processing...")
    
    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_screen_points = params["screen_points"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]
     
    init_opacity = params["opacity"]
    # throw away low opacity kernels
    mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
    print(">>> No. of particles before OPACITY filtering", init_pos.shape[0])
    init_pos = init_pos[mask, :]
    init_cov = init_cov[mask, :]
    init_opacity = init_opacity[mask, :]
    init_screen_points = init_screen_points[mask, :]
    init_shs = init_shs[mask, :]
    print(">>> No. of particles after OPACITY filtering", init_pos.shape[0])

    print("INIT_SHS.shape", init_shs.shape)

    # rorate and translate object
    if args.debug:
        if not os.path.exists("./log"):
            os.makedirs("./log")
        particle_position_tensor_to_ply(
            init_pos,
            "./log/init_particles.ply",
        )
    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"],
    )
    print(">>>> INIT_POS", init_pos.shape)
    print("max of init_pos", init_pos.max(axis=0))
    print("min of init_pos", init_pos.min(axis=0))
    rotated_pos = apply_rotations(init_pos, rotation_matrices)

    if args.debug:
        particle_position_tensor_to_ply(rotated_pos, "./log/rotated_particles.ply")

    # select a sim area and save params of unslected particles
    unselected_pos, unselected_cov, unselected_opacity, unselected_shs = (
        None,
        None,
        None,
        None,
    )
    if preprocessing_params["sim_area"] is not None:
        boundary = preprocessing_params["sim_area"]
        assert len(boundary) == 6
        mask = torch.ones(rotated_pos.shape[0], dtype=torch.bool).to(device="cuda")
        for i in range(3):
            mask = torch.logical_and(mask, rotated_pos[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, rotated_pos[:, i] < boundary[2 * i + 1])

        unselected_pos = init_pos[~mask, :]
        unselected_cov = init_cov[~mask, :]
        unselected_opacity = init_opacity[~mask, :]
        unselected_shs = init_shs[~mask, :]

        rotated_pos = rotated_pos[mask, :]
        init_cov = init_cov[mask, :]
        init_opacity = init_opacity[mask, :]
        init_shs = init_shs[mask, :]

    transformed_pos, scale_origin, original_mean_pos = transform2origin(rotated_pos)
    print(">>>> AFTER TRANSFORM2ORIGIN")
    print("min of transformed_pos", transformed_pos.min(axis=0))
    print("max of transformed_pos", transformed_pos.max(axis=0))
    transformed_pos = shift2center111(transformed_pos)
    print(">>>> AFTER SHIFT2CENTER111")
    print("min of transformed_pos", transformed_pos.min(axis=0))
    print("max of transformed_pos", transformed_pos.max(axis=0))

    # modify covariance matrix accordingly
    init_cov = apply_cov_rotations(init_cov, rotation_matrices)
    init_cov = scale_origin * scale_origin * init_cov

    if args.debug:
        particle_position_tensor_to_ply(
            transformed_pos,
            "./log/transformed_particles.ply",
        )

    # fill particles if needed
    gs_num = transformed_pos.shape[0]
    device = "cuda:0"
    filling_params = preprocessing_params["particle_filling"]

    if filling_params is not None:
        print("Filling internal particles...")
        mpm_init_pos = fill_particles(
            pos=transformed_pos,
            opacity=init_opacity,
            cov=init_cov,
            grid_n=filling_params["n_grid"],
            max_samples=filling_params["max_particles_num"],
            grid_dx=material_params["grid_lim"] / filling_params["n_grid"],
            density_thres=filling_params["density_threshold"],
            search_thres=filling_params["search_threshold"],
            max_particles_per_cell=filling_params["max_partciels_per_cell"],
            search_exclude_dir=filling_params["search_exclude_direction"],
            ray_cast_dir=filling_params["ray_cast_direction"],
            boundary=filling_params["boundary"],
            smooth=filling_params["smooth"],
        ).to(device=device)

        if args.debug:
            particle_position_tensor_to_ply(mpm_init_pos, "./log/filled_particles.ply")
    else:
        mpm_init_pos = transformed_pos.to(device=device)

    # init the mpm solver
    print("Initializing MPM solver and setting up boundary conditions...")
    print("\n=== CHECKING PARTICLE POSITIONS FOR MPM SOLVER ===")
    print(f"Particle range: [{mpm_init_pos.min(dim=0)[0][0]:.4f}, {mpm_init_pos.max(dim=0)[0][0]:.4f}] x "
          f"[{mpm_init_pos.min(dim=0)[0][1]:.4f}, {mpm_init_pos.max(dim=0)[0][1]:.4f}] x "
          f"[{mpm_init_pos.min(dim=0)[0][2]:.4f}, {mpm_init_pos.max(dim=0)[0][2]:.4f}]")


    # Check if particles are too close to the grid boundary
    grid_lim = material_params["grid_lim"]
    margin = 0.05  # Safety margin
    too_close_to_min = (mpm_init_pos < margin).any(dim=1).sum().item()
    too_close_to_max = (mpm_init_pos > grid_lim - margin).any(dim=1).sum().item()

    if too_close_to_min > 0 or too_close_to_max > 0:
        print(f"WARNING: {too_close_to_min} particles are too close to the minimum boundary")
        print(f"WARNING: {too_close_to_max} particles are too close to the maximum boundary")
        print("This may cause simulation instability if particles move outside the grid")
        
        # Optional: Adjust grid_lim to provide more space
        if material_params["grid_lim"] < 2.5:
            print(f"Increasing grid_lim from {material_params['grid_lim']} to 2.5 for stability")
            material_params["grid_lim"] = 2.5
            print(f"New valid range: [0, {material_params['grid_lim']}]")

    print("=== END CHECKING ===\n")

    mpm_init_vol = get_particle_volume(
        mpm_init_pos,
        material_params["n_grid"],
        material_params["grid_lim"] / material_params["n_grid"],
        unifrom=material_params["material"] == "sand",
    ).to(device=device)

    if filling_params is not None and filling_params["visualize"] == True:
        shs, opacity, mpm_init_cov = init_filled_particles(
            mpm_init_pos[:gs_num],
            init_shs,
            init_cov,
            init_opacity,
            mpm_init_pos[gs_num:],
        )
        gs_num = mpm_init_pos.shape[0]
    else:
        mpm_init_cov = torch.zeros((mpm_init_pos.shape[0], 6), device=device)
        mpm_init_cov[:gs_num] = init_cov
        shs = init_shs
        opacity = init_opacity

    if args.debug:
        print("check *.ply files to see if it's ready for simulation")


    print("mpm_init_pos", mpm_init_pos.shape) ## (num_particles, 3)
    ## we constantly run into the error that the "some particles are out of the simulation grids. "
    ### Let's check the simulation grid size + the particles' position
    print("material_params['n_grid']", material_params["n_grid"])
    print("material_params['grid_lim']", material_params["grid_lim"])
    ### get the min and max of the particles' position
    print("mpm_init_pos.min()", mpm_init_pos.min(axis=0))
    print("mpm_init_pos.max()", mpm_init_pos.max(axis=0))

    # set up the mpm solver
    mpm_solver = MPM_Simulator_WARP(10)
    mpm_solver.load_initial_data_from_torch(
        mpm_init_pos,
        mpm_init_vol,
        mpm_init_cov,
        n_grid=material_params["n_grid"],
        grid_lim=material_params["grid_lim"],
    )
    mpm_solver.set_parameters_dict(material_params)

    # Note: boundary conditions may depend on mass, so the order cannot be changed!
    set_boundary_conditions(mpm_solver, bc_params, time_params)

    if args.similarity_path and os.path.exists(args.similarity_path):
        print(f"Applying similarity-based materials from {args.similarity_path}")

        ### For now just assuming there are two materials, light (soft) and stiff.
        ### dense_similarities.npy contains {0, 1} values, 1 meaning soft material and 0 stiff. This input could easily be [0,1] and
        ### the rest of the code should work.
        ### apply_similarity_based_materials_to_simulation assigns these material values.
        
        light_material = {
            "E": 5e3,           # Softer 
            "nu": 0.4,          # Same as original
            "density": 10.0     # Use the lighter density
        }
        
        stiff_material = {
            "E": 2e7,     
            "nu": 0.3,        
            "density": 400.0    
        }
        
        # Apply the material properties based on similarity values
        apply_similarity_based_materials_to_simulation(
            mpm_solver=mpm_solver,
            similarity_path=args.similarity_path,
            light_material=light_material,
            stiff_material=stiff_material,
            device=device
        )
    else:
        print("No similarity data found, using uniform material properties")
        mpm_solver.finalize_mu_lam()
        
    # camera setting
    mpm_space_viewpoint_center = (
        torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    )
    mpm_space_vertical_upward_axis = (
        torch.tensor(camera_params["mpm_space_vertical_upward_axis"])
        .reshape((1, 3))
        .cuda()
    )
    (
        viewpoint_center_worldspace,
        observant_coordinates,
    ) = get_center_view_worldspace_and_observant_coordinate(
        mpm_space_viewpoint_center,
        mpm_space_vertical_upward_axis,
        rotation_matrices,
        scale_origin,
        original_mean_pos,
    )

    # run the simulation
    if args.output_ply or args.output_h5:
        directory_to_save = os.path.join(args.output_path, "simulation_ply")
        if not os.path.exists(directory_to_save):
            os.makedirs(directory_to_save)

        save_data_at_frame(
            mpm_solver,
            directory_to_save,
            0,
            save_to_ply=args.output_ply,
            save_to_h5=args.output_h5,
        )

    substep_dt = time_params["substep_dt"]
    frame_dt = time_params["frame_dt"]
    frame_num = time_params["frame_num"]
    step_per_frame = int(frame_dt / substep_dt)
    opacity_render = opacity
    shs_render = shs
    height = None
    width = None
    
    # frame_num = 1
    for frame in tqdm(range(frame_num)):
        for step in range(step_per_frame):
            mpm_solver.p2g2p(frame, substep_dt, device=device)
            
            # Add this check every 10 steps to avoid slowing down the simulation too much
            if step % 10 == 0 and args.debug:
                # Check if any particles have moved outside the grid
                pos = mpm_solver.export_particle_x_to_torch()
                outside_grid = ((pos < 0) | (pos > material_params["grid_lim"])).any(dim=1).sum().item()
                if outside_grid > 0:
                    print(f"WARNING: {outside_grid} particles have moved outside the grid at frame {frame}, step {step}")
                    # Clamp particles to keep them in bounds
                    pos_clamped = torch.clamp(pos, min=0.01, max=material_params["grid_lim"]-0.01)
                    # Update particle positions in the solver
                    mpm_solver.load_particle_x_from_torch(pos_clamped)
                    print("Particles have been clamped to stay within grid bounds")

        if args.output_ply or args.output_h5:
            save_data_at_frame(
                mpm_solver,
                directory_to_save,
                frame + 1,
                save_to_ply=args.output_ply,
                save_to_h5=args.output_h5,
            )

        if args.render_img:
            pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
            
            # Transform positions back to original coordinate system
            pos = apply_inverse_rotations(
                undotransform2origin(
                    undoshift2center111(pos), scale_origin, original_mean_pos
                ),
                rotation_matrices,
            )
            
            # Add unselected particles if needed
            if preprocessing_params["sim_area"] is not None:
                pos = torch.cat([pos, unselected_pos], dim=0)
            
            # Use the simplified visualization function
            height, width = save_point_cloud_image(
                pos,
                args.output_path,
                frame,
                color=params['colors'].cpu().numpy() if params['colors'] is not None else None,
                padding=0.2,  # 20% padding around the object
                elevation=20,  # Lower elevation to see more from above
                azimuth=30,   # Adjust view angle
                dpi=200,
                point_size=3  # Slightly larger points for better visibility
            )
            
    if args.render_img and args.compile_video:
        # Get all saved PNG frames
        frame_files = sorted(glob.glob(os.path.join(args.output_path, '*.png')))
        if len(frame_files) == 0:
            print("No PNG frames found in", args.output_path)
        else:
            # Read a sample frame to determine dimensions
            sample_frame = cv2.imread(frame_files[0])
            height, width, _ = sample_frame.shape

            # Set frames per second
            fps = int(1.0 / time_params["frame_dt"])
            
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = os.path.join(args.output_path, 'output.mp4')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            # Write each frame to the video
            for frame_file in frame_files:
                frame = cv2.imread(frame_file)
                if frame is None:
                    print(f"Warning: could not read frame {frame_file}")
                    continue
                video_writer.write(frame)
            
            video_writer.release()
            print("Video successfully saved to:", video_path)
            
            # Also create a GIF from the frames
            try:
                from PIL import Image
                
                # Create GIF
                gif_path = os.path.join(args.output_path, 'output.gif')
                frames = []
                
                # Determine GIF frame duration in milliseconds
                duration = int(1000 / fps)  # Convert fps to milliseconds per frame
                
                # Load all frames
                for frame_file in frame_files:
                    img = Image.open(frame_file)
                    frames.append(img.copy())
                    
                # Save as GIF
                frames[0].save(
                    gif_path,
                    format='GIF',
                    append_images=frames[1:],
                    save_all=True,
                    duration=duration,
                    loop=0  # 0 means loop forever
                )
                print("GIF successfully saved to:", gif_path)
            except ImportError:
                print("PIL library not found. GIF creation skipped.")
            except Exception as e:
                print(f"Error creating GIF: {e}")