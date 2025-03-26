import argparse
import os
from utils import on_desktop
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_id", type=str, help="Objaverse object ID to process")
    parser.add_argument("--num_images", type=int, help="Number of images to render", default=100)
    parser.add_argument("--train_steps", type=int, help="Number of training steps", default=30_000)
    args = parser.parse_args()

    is_on_desktop = on_desktop()
    path_prefix = "/mnt/kostas-graid/datasets/vlongle/diffphys3d" if not is_on_desktop else "."

    download_cmd = f"python download_objaverse.py --obj_id {args.obj_id}"
    blender_render_cmd = f'blender --background --python generate_blendernerf_data.py -- --obj_id {args.obj_id} --num_images {args.num_images} --format NGP --camera_dist 1.8 --output_dir {path_prefix}/data'
    if not is_on_desktop:
        blender_render_cmd = f'export PATH="/mnt/kostas-graid/sw/envs/vlongle/blender/blender-4.3.2-linux-x64:$PATH"; {blender_render_cmd}'
    convert_cmd = f"python convert.py --obj_id {args.obj_id}"
    train_cmd = f"ns-train f3rm --data {path_prefix}/data/{args.obj_id} --max-num-iterations {args.train_steps} --viewer.quit-on-train-completion True"

    os.system(download_cmd)
    os.system(blender_render_cmd)
    os.system(convert_cmd)
    os.system(train_cmd)
    
    #  # Find the latest config file in the output directory
    output_dir = f"outputs/{args.obj_id}/f3rm"
    latest_run = max([os.path.join(output_dir, d) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))], key=os.path.getmtime)
    config_path = os.path.join(latest_run, "config.yml")
    
    render_output_dir = f"render_outputs/{args.obj_id}"
    ns_render_cmd = f"ns-render dataset --load-config {config_path} --output-path {render_output_dir} --split=train --rendered_output_names=rgb"
    os.system(ns_render_cmd)


    voxel_cmd = f"python voxel_to_pc.py --scene {config_path} --output {render_output_dir}/clip_features.npz"

    os.system(voxel_cmd)

    voxel_pc_cmd = voxel_cmd + " --extract_pc"

    os.system(voxel_pc_cmd)

