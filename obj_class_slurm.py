import os
import glob
from pathlib import Path
import subprocess
import argparse
import json

def find_objaverse_assets(base_dir):
    """
    Walk through the directory structure to find all Objaverse assets.
    
    Args:
        base_dir (str): Base directory to search in
    
    Returns:
        list: List of tuples containing (obj_id, object_path)
    """
    results = []
    
    # Ensure the base directory exists
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist")
        return results
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_dir):
        # Look for .glb files
        for file in files:
            if file.endswith('.glb'):
                # Extract the object ID from the filename (remove extension)
                obj_id = os.path.splitext(file)[0]
                object_path = os.path.join(root, file)
                results.append((obj_id, object_path))
    
    print(f"Found {len(results)} Objaverse assets")
    return results

def main():
    parser = argparse.ArgumentParser(description='Process Objaverse assets using Slurm job arrays')
    parser.add_argument('--base_dir', type=str, default="/mnt/kostas-graid/datasets/vlongle/diffphys3d/assets/tree/",
                        help='Base directory to search for assets')
    parser.add_argument('--camera_dist_min', type=float, default=1.2, help='Minimum camera distance')
    parser.add_argument('--camera_dist_max', type=float, default=1.8, help='Maximum camera distance')
    parser.add_argument('--scene_scale', type=float, default=1.0, help='Scene scale')
    parser.add_argument('--num_images', type=int, default=200, help='Number of images to render')
    parser.add_argument('--time', type=str, default="24:00:00", help='Job time limit (HH:MM:SS)')
    parser.add_argument('--partition', type=str, default="eaton-compute", help='Slurm partition')
    parser.add_argument('--qos', type=str, default="ee-high", help='Quality of service')
    parser.add_argument('--gpu', type=str, default="1", help='GPU resource request')
    parser.add_argument('--mem', type=str, default="32G", help='Memory request')
    parser.add_argument('--cpus', type=int, default=8, help='CPUs per task')
    parser.add_argument('--limit', type=int, help='Limit the number of assets to process')
    
    args = parser.parse_args()
    
    # Find all assets
    assets = find_objaverse_assets(args.base_dir)
    
    # Limit the number of assets if specified
    if args.limit and args.limit < len(assets):
        assets = assets[:args.limit]
        print(f"Limited to {args.limit} assets")
    
    # Create temporary files with object IDs and paths
    os.makedirs('slurm_data', exist_ok=True)
    
    obj_ids_file = 'slurm_data/obj_ids.txt'
    obj_paths_file = 'slurm_data/obj_paths.txt'
    
    with open(obj_ids_file, 'w') as f_ids, open(obj_paths_file, 'w') as f_paths:
        for obj_id, path in assets:
            f_ids.write(f"{obj_id}\n")
            f_paths.write(f"{path}\n")
    
    print(f"Wrote {len(assets)} object IDs to {obj_ids_file}")
    print(f"Wrote {len(assets)} object paths to {obj_paths_file}")
    
    # Import the function here to avoid circular imports
    from generate_slurm_job import generate_slurm_job_array
    
    # Extract object IDs and paths from assets
    obj_ids = [asset[0] for asset in assets]
    obj_paths = [asset[1] for asset in assets]
    # obj_ids = obj_ids[:10]
    # obj_paths = obj_paths[:10]
    
    # Generate and optionally submit the job array
    result = generate_slurm_job_array(
        obj_ids=obj_ids,
        obj_paths=obj_paths,
        camera_dist_min=args.camera_dist_min,
        camera_dist_max=args.camera_dist_max,
        scene_scale=args.scene_scale,
        num_images=args.num_images,
        time=args.time,
        partition=args.partition,
        qos=args.qos,
        gpu=args.gpu,
        mem=args.mem,
        cpus=args.cpus,
        submit=True,
        array_limit=len(obj_ids),
    )
    

if __name__ == "__main__":
    main()