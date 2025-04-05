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
from material_field import apply_material_field_to_simulation

# Utils
from utils.decode_param import *
from utils.transformation_utils import *
from utils.camera_view_utils import *
from utils.render_utils import *
import glob
from gs_simulation_pc import load_point_cloud

wp.init()
wp.config.verify_cuda = True

ti.init(arch=ti.cuda, device_memory_GB=8.0)


class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def load_checkpoint(model_path, sh_degree=3, iteration=-1):
    # Find checkpoint
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(
        checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
    )

    # Load guassians
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
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

    if not os.path.exists(args.model_path):
        AssertionError("Model path does not exist!")
    if not os.path.exists(args.config):
        AssertionError("Scene config does not exist!")
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

    # load gaussians
    print("Loading gaussians...")
    model_path = args.model_path
    gaussians = load_checkpoint(model_path)
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        if args.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    )

    # init the scene
    print("Initializing scene and pre-processing...")
    params = load_params_from_gs(gaussians, pipeline)

    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_screen_points = params["screen_points"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]

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
    # mpm_solver.set_parameters_dict(material_params)

    # Note: boundary conditions may depend on mass, so the order cannot be changed!
    set_boundary_conditions(mpm_solver, bc_params, time_params)


    pc_params = load_point_cloud(args.point_cloud_path)
    apply_material_field_to_simulation(mpm_solver, pc_params, device=device,
                                       scale_origin=scale_origin, original_mean_pos=original_mean_pos, rotation_matrices=rotation_matrices)

    # mpm_solver.finalize_mu_lam()

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
    for frame in tqdm(range(frame_num)):
        current_camera = get_camera_view(
            model_path,
            default_camera_index=camera_params["default_camera_index"],
            center_view_world_space=viewpoint_center_worldspace,
            observant_coordinates=observant_coordinates,
            show_hint=camera_params["show_hint"],
            init_azimuthm=camera_params["init_azimuthm"],
            init_elevation=camera_params["init_elevation"],
            init_radius=camera_params["init_radius"],
            move_camera=camera_params["move_camera"],
            current_frame=frame,
            delta_a=camera_params["delta_a"],
            delta_e=camera_params["delta_e"],
            delta_r=camera_params["delta_r"],
        )
        rasterize = initialize_resterize(
            current_camera, gaussians, pipeline, background
        )

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
                    mpm_solver.import_particle_x_from_torch(pos_clamped)
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
            cov3D = mpm_solver.export_particle_cov_to_torch()
            rot = mpm_solver.export_particle_R_to_torch()
            cov3D = cov3D.view(-1, 6)[:gs_num].to(device)
            rot = rot.view(-1, 3, 3)[:gs_num].to(device)

            pos = apply_inverse_rotations(
                undotransform2origin(
                    undoshift2center111(pos), scale_origin, original_mean_pos
                ),
                rotation_matrices,
            )
            cov3D = cov3D / (scale_origin * scale_origin)
            cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
            opacity = opacity_render
            shs = shs_render
            if preprocessing_params["sim_area"] is not None:
                pos = torch.cat([pos, unselected_pos], dim=0)
                cov3D = torch.cat([cov3D, unselected_cov], dim=0)
                opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
                shs = torch.cat([shs_render, unselected_shs], dim=0)

            colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)
            rendering, raddi = rasterize(
                means3D=pos,
                means2D=init_screen_points,
                shs=None,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D,
            )
            cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            if height is None or width is None:
                height = cv2_img.shape[0] // 2 * 2
                width = cv2_img.shape[1] // 2 * 2
            assert args.output_path is not None
            cv2.imwrite(
                os.path.join(args.output_path, f"{frame}.png".rjust(8, "0")),
                255 * cv2_img,
            )

    # if args.render_img and args.compile_video:
    #     fps = int(1.0 / time_params["frame_dt"])
    #     os.system(
    #         f"ffmpeg -framerate {fps} -i {args.output_path}/%04d.png -c:v libx264 -s {width}x{height} -y -pix_fmt yuv420p {args.output_path}/output.mp4"
    #     )

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

            # Also create a GIF from the frames
            try:
                from PIL import Image
                
                # Create GIF
                gif_path = os.path.join(args.output_path, 'output.gif')
                frames = []
                
                # Determine GIF frame duration in milliseconds
                # Lower duration = faster animation
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

