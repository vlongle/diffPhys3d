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
    parser.add_argument("--only_normalize", type=str2bool, help="Only normalize the scene", default=False)
    parser.add_argument("--voxel_size", type=float, help="Voxel size", default=0.01)
    parser.add_argument("--part_queries", type=str, help="Part queries", default="pot, trunk, leaves")
    args = parser.parse_args()

    start_time = time.time()

    is_on_desktop = on_desktop()
    path_prefix = "/mnt/kostas-graid/datasets/vlongle/diffphys3d" if not is_on_desktop else os.getcwd()

    # Only download if obj_path is not provided
    if args.obj_path is None:
        download_cmd = f"python download_objaverse.py --obj_id {args.obj_id}"
        os.system(download_cmd)
        os.system(download_cmd)
    
    # Use obj_path if provided, otherwise use obj_id
    obj_param = f"--obj_path {args.obj_path}" if args.obj_path is not None else f"--obj_id {args.obj_id}"
    
    blender_render_cmd = f'blender --background --python generate_blendernerf_data.py -- {obj_param} --num_images {args.num_images} --format NGP --camera_dist_min {args.camera_dist_min} --camera_dist_max {args.camera_dist_max} --output_dir {path_prefix}/data'
    if args.transparent_bg:
        blender_render_cmd += ' --transparent_bg'
    if not is_on_desktop:
        blender_render_cmd = f'export PATH="/mnt/kostas-graid/sw/envs/vlongle/blender/blender-4.3.2-linux-x64:$PATH"; {blender_render_cmd}'
    blender_render_cmd += f" --scene_scale {args.scene_scale}"
    if args.only_normalize:
        blender_render_cmd += " --only_normalize"

    # Use obj_id for the rest of the pipeline regardless of input method
    convert_cmd = f"python convert.py --obj_id {args.obj_id} --data_dir {path_prefix}/data"
    
    method = "f3rm"
    # method = "nerfacto"
    train_cmd = f"ns-train {method} --data {path_prefix}/data/{args.obj_id} --max-num-iterations {args.train_steps} --viewer.quit-on-train-completion True --save_only_latest_checkpoint False --output_dir {path_prefix}/outputs"


    # os.system(blender_render_cmd)
    # os.system(convert_cmd)
    # os.system(train_cmd)
    
    # # #  # Find the latest config file in the output directory
    output_dir = f"{path_prefix}/outputs/{args.obj_id}/{method}"
    # output_dir = f"{path_prefix}/outputs/{args.obj_id}_cam_1/f3rm"
    latest_run = max([os.path.join(output_dir, d) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))], key=os.path.getmtime)
    config_path = os.path.join(latest_run, "config.yml")
    
    render_output_dir = f"{path_prefix}/render_outputs/{args.obj_id}"
    ns_render_cmd = f"ns-render dataset --load-config {config_path} --output-path {render_output_dir} --split=train --rendered_output_names=rgb"
    # os.system(ns_render_cmd)

    voxel_cmd = f"python voxel_to_pc.py --scene {config_path} --output {render_output_dir}/clip_features.npz --voxel_size {args.voxel_size}"
    # os.system(voxel_cmd)

    voxel_pc_cmd = voxel_cmd + " --extract_pc"
    # os.system(voxel_pc_cmd)

    
    segmentation_cmd = f"python segmentation.py --grid_feature_path {render_output_dir}/clip_features.npz --occupancy_path {render_output_dir}/clip_features_pc.ply --output_dir {render_output_dir} --part_queries '{args.part_queries}' --material_dict_path {render_output_dir}/material_dict.json --use_spatial_smoothing True"
    print(">> SEGMENTATION CMD: ", segmentation_cmd)
    os.system(segmentation_cmd)


    material_field = "segmented_rgb.ply"
    ## TODO: still some bugs with the custom config vs cuboid config...
    # phys_config = "custom_cuboid_config.json"
    phys_config = "custom_config.json"
    phys_sim_cmd = f"cd third_party/PhysGaussian; xvfb-run -a  python gs_simulation_pc.py --point_cloud_path {render_output_dir}/{material_field} --output_path nerf_pc_ununiform_custom_output --config ./config/{phys_config} --render_img --compile_video --white_bg --debug"
    os.system(phys_sim_cmd)

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")
