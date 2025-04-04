import argparse
import os
from utils import on_desktop
import time
from utils import str2bool

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_id", type=str, help="Objaverse object ID to process", required=True)
    parser.add_argument("--obj_path", type=str, help="Path to the object file to process", default=None)
    parser.add_argument("--num_images", type=int, help="Number of images to render", default=200)
    parser.add_argument("--train_steps", type=int, help="Number of training steps", default=5_000)
    # parser.add_argument("--train_steps", type=int, help="Number of training steps", default=10_000)
    parser.add_argument("--camera_dist_min", type=float, help="Minimum camera distance", default=1.2)
    parser.add_argument("--camera_dist_max", type=float, help="Maximum camera distance", default=1.8)
    # parser.add_argument("--camera_dist", type=float, help="Camera distance (deprecated, use min/max instead)", default=1.2)
    parser.add_argument("--transparent_bg", type=str2bool, help="Use transparent background", default=True)
    parser.add_argument("--scene_scale", type=float, help="Scene scale", default=1.0)
    parser.add_argument("--voxel_size", type=float, help="Voxel size", default=0.01)
    # parser.add_argument("--part_queries", type=str, help="Part queries", default="pot, trunk, leaves")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing data")
    args = parser.parse_args()

    start_time = time.time()

    is_on_desktop = on_desktop()
    path_prefix = "/mnt/kostas-graid/datasets/vlongle/diffphys3d" if not is_on_desktop else os.getcwd()

    normal_env = "diffphys3d"
    gs_env = "gaussian_splatting"
    
    # Updated conda activation commands
    conda_init = "source $(conda info --base)/etc/profile.d/conda.sh"
    normal_acv = f"bash -c '{conda_init} && conda activate {normal_env} &&"
    gs_acv = f"bash -c '{conda_init} && conda activate {gs_env} &&"


    # Only download if obj_path is not provided
    if args.obj_path is None:
        download_cmd = f"{normal_acv} python download_objaverse.py --obj_id {args.obj_id}'"
        os.system(download_cmd)
    
    # Use obj_path if provided, otherwise use obj_id
    obj_param = f"--obj_path {args.obj_path}" if args.obj_path is not None else f"--obj_id {args.obj_id}"
    
    blender_render_cmd = f'blender --background --python generate_blendernerf_data.py -- {obj_param} --num_images {args.num_images} --format NGP --camera_dist_min {args.camera_dist_min} --camera_dist_max {args.camera_dist_max} --output_dir {path_prefix}/data'
    if args.transparent_bg:
        blender_render_cmd += ' --transparent_bg'
    if not is_on_desktop:
        blender_render_cmd = f'export PATH="/mnt/kostas-graid/sw/envs/vlongle/blender/blender-4.3.2-linux-x64:$PATH"; {blender_render_cmd}'
    blender_render_cmd += f" --scene_scale {args.scene_scale}"

    # Use obj_id for the rest of the pipeline regardless of input method
    convert_cmd = f"{normal_acv} python convert.py --obj_id {args.obj_id} --data_dir {path_prefix}/data'"
    
    method = "f3rm"
    # method = "nerfacto"
    train_cmd = f"{normal_acv} ns-train {method} --data {path_prefix}/data/{args.obj_id} --max-num-iterations {args.train_steps} --viewer.quit-on-train-completion True --save_only_latest_checkpoint False --output_dir {path_prefix}/outputs'"

    gs_train_cmd = f"{gs_acv} cd third_party/gaussian-splatting && python train.py -s {path_prefix}/data/{args.obj_id} --iterations {args.train_steps} --model_path {path_prefix}/outputs/{args.obj_id}/gs'"
    print(">> GS TRAIN CMD: ", gs_train_cmd)


    print(">> BLENDER RENDER CMD: ", blender_render_cmd, "path: ", f"{path_prefix}/data/{args.obj_id}/transforms_train.json")
    if args.overwrite or not os.path.exists(f"{path_prefix}/data/{args.obj_id}/transforms_train.json"):
        os.system(blender_render_cmd)
    os.system(convert_cmd)

    # if args.overwrite or not os.path.exists(f"{path_prefix}/outputs/{args.obj_id}/gs"):
    #     os.system(train_cmd)
    #     os.system(gs_train_cmd)

    # # #  # Find the latest config file in the output directory
    output_dir = f"{path_prefix}/outputs/{args.obj_id}/{method}"
    # output_dir = f"{path_prefix}/outputs/{args.obj_id}_cam_1/f3rm"
    latest_run = max([os.path.join(output_dir, d) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))], key=os.path.getmtime)
    config_path = os.path.join(latest_run, "config.yml")
    
    render_output_dir = f"{path_prefix}/render_outputs/{args.obj_id}"
    ns_render_cmd = f"{normal_acv} ns-render dataset --load-config {config_path} --output-path {render_output_dir} --split=train --rendered_output_names=rgb"
    # os.system(ns_render_cmd)

    voxel_cmd = f"{normal_acv} python voxel_to_pc.py --scene {config_path} --output {render_output_dir}/clip_features.npz --voxel_size {args.voxel_size}'"

    if args.overwrite or not os.path.exists(f"{render_output_dir}/clip_features.npz"):
        os.system(voxel_cmd)
    
    segmentation_cmd = f"{normal_acv} python segmentation.py --grid_feature_path {render_output_dir}/clip_features.npz --occupancy_path {render_output_dir}/clip_features_pc.ply --output_dir {render_output_dir}  --material_dict_path {render_output_dir}/material_dict.json --use_spatial_smoothing True --overwrite {args.overwrite}'"
    print(">> SEGMENTATION CMD: ", segmentation_cmd)
    os.system(segmentation_cmd)


    # material_field = "segmented_rgb.ply"
    material_field = "segmented_semantics.ply"
    ## TODO: still some bugs with the custom config vs cuboid config...
    # phys_config = "custom_cuboid_config.json"

    sim_out_path = f"{render_output_dir}/sim_output"
    phys_config = "custom_config.json"
    phys_sim_cmd = f"{normal_acv} cd third_party/PhysGaussian && xvfb-run -a python gs_simulation_pc.py --point_cloud_path {render_output_dir}/{material_field} --output_path {sim_out_path} --config ./config/{phys_config} --render_img --compile_video --white_bg --debug'"
    gs_sim_cmd =   f"{normal_acv} cd third_party/PhysGaussian && xvfb-run -a python gs_simulation.py --model_path {path_prefix}/outputs/{args.obj_id}/gs --point_cloud_path {render_output_dir}/{material_field} --output_path {sim_out_path} --config ./config/{phys_config} --render_img --compile_video --white_bg --debug'"
    if not is_on_desktop:
        phys_sim_cmd = f"export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6; {phys_sim_cmd}"
        gs_sim_cmd = f"export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6; {gs_sim_cmd}"
    print(">> PHYS SIM CMD: ", phys_sim_cmd)
    # os.system(phys_sim_cmd)


    os.system(gs_sim_cmd)

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")
