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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

import os
import trimesh
import numpy as np


def visualize_point_cloud_trimesh(positions, output_path, frame_number, camera_params=None, color=None, 
                                  camera_transform=None):
    """
    Visualize a point cloud using trimesh with a consistent camera view across frames.
    
    Args:
        positions: Tensor of 3D point positions [N, 3]
        output_path: Directory to save the rendered image
        frame_number: Current frame number for filename
        camera_params: Dictionary of camera parameters to ensure consistency across frames
        color: Optional color array for points
        point_size: Size of points in the plot
        show_axes: Whether to display coordinate axes
    """
    # Convert positions to numpy
    positions_np = positions.cpu().numpy()
    
    # Create a PointCloud object
    point_cloud = trimesh.PointCloud(positions_np)
    
    # Handle colors
    if color is not None:
        if isinstance(color, np.ndarray):
            point_cloud.colors = color
        else:
            if isinstance(color, (list, tuple)) and len(color) == 3:
                if max(color) <= 1.0:
                    color = [int(c * 255) for c in color]
                colors = np.tile(color, (positions_np.shape[0], 1))
                point_cloud.colors = colors
    
    # Create a scene and add the point cloud
    scene = trimesh.Scene()
    scene.add_geometry(point_cloud)
    
    
    camera_params = {
            'fov': (60, 60),
            'resolution': [1000, 1000],
            'center': [0, 0, 0],
            'distance': 1.0,
            'angle': np.deg2rad([90, 0, 180])  # [elevation, azimuth, roll]
        }
    
    # Set camera using the provided or default parameters
    scene.set_camera(angles=camera_params['angle'],
                     resolution=camera_params['resolution'],
                     fov=camera_params['fov'])
    
    if camera_transform is not None:
        scene.camera_transform = camera_transform
    # Render the image
    rendered = scene.save_image(resolution=camera_params['resolution'],
                                visible=True,
                                flags={'auto_view': False})
    
    # Save the image
    output_file = os.path.join(output_path, f"{frame_number:04d}.png")
    with open(output_file, 'wb') as f:
        f.write(rendered)
    
    # Return dimensions of the saved image
    return camera_params['resolution'][0], camera_params['resolution'][1], scene.camera_transform



def visualize_point_cloud(positions, output_path, frame_number, color=None, point_size=1,
                          zoom_factor=1.0):
    """
    Visualize a point cloud using matplotlib's 3D scatter plot.
    
    Args:
        positions: Tensor of 3D point positions [N, 3]
        output_path: Directory to save the rendered image
        frame_number: Current frame number for filename
        color: Optional color array for points
        point_size: Size of points in the plot
    """
    # Convert positions to numpy
    positions_np = positions.cpu().numpy()

    ## save positions_np to a ply file
    # np.save(os.path.join(output_path, f"{frame_number:04d}_positions.npy"), positions_np)
    
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(positions_np[:, 0], positions_np[:, 1], positions_np[:, 2], s=point_size, 
               c=color)
    # Set consistent view angle
    ax.view_init(elev=0, azim=90)

    limit = 0.5 * zoom_factor
    x_min, x_max = -limit, limit
    y_min, y_max = -limit, limit
    z_min, z_max = -limit, limit


    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)


    ax.set_axis_off()

    
    # Save figure
    output_file = os.path.join(output_path, f"{frame_number:04d}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # print(f"Rendered frame {frame_number} with {positions_np.shape[0]} points")
    
    # Return dimensions of the saved image
    return 1000, 1000  # Approximate size based on figsize and dpi


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

    # exit(0)
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
    # scale_factor = 0.005  # e.g. reduce the size of the entire gaussian splatting scene to 1% of original
    # mpm_init_pos *= scale_factor


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

    
    camera_transform=None



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
            
            # Visualize point cloud
            # height, width = visualize_point_cloud(
            #     pos, 
            #     args.output_path, 
            #     frame,
            #     color=params['colors'].cpu().numpy(),
            #     point_size=2,  # Adjust point size as needed,
            #     zoom_factor=0.5,
            # )


            
            ## NOTE: TODO: the visualize_point_cloud with matplotlib is correct but
            ## the result looks kinda sparse and shitty. trimesh looks better but
            ## somehow the pot is also swaying, which IS NOT CORRECT.
            ## I think point_cloud_trimesh actually renders out a voxel-ish looking
            ## thing, which is nicer than just point projection like the matplotlib one.
            height, width, camera_transform = visualize_point_cloud_trimesh(
                pos,
                args.output_path,
                frame,
                color=params['colors'].cpu().numpy(),
                camera_transform=camera_transform
            )
            
            
    if args.render_img and args.compile_video:
        # Get all saved PNG frames (assuming they are named as 0000.png, 0001.png, etc.)
        frame_files = sorted(glob.glob(os.path.join(args.output_path, '*.png')))
        if len(frame_files) == 0:
            print("No PNG frames found in", args.output_path)
        else:
            # Read a sample frame to determine dimensions
            sample_frame = cv2.imread(frame_files[0])
            height, width, _ = sample_frame.shape

            # Set frames per second; here we use frame_dt from time_params
            fps = int(1.0 / time_params["frame_dt"])
            
            # Define the codec and create VideoWriter object
            # 'mp4v' is widely supported; adjust if necessary
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

