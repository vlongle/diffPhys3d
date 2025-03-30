import os
import argparse
import subprocess

def generate_slurm_job(obj_id, camera_dist_min, camera_dist_max, scene_scale, num_images, 
                       time="24:00:00", partition="batch", qos="normal", 
                       gpu="1", mem="32G", cpus=8):
    """Generate and submit a Slurm job to run the 3D object processing pipeline."""
    
    os.makedirs('slurm_scripts', exist_ok=True)
    os.makedirs('slurm_outs', exist_ok=True)
    
    # Handle both single object ID and list of object IDs
    obj_ids = [obj_id] if isinstance(obj_id, str) else obj_id
    
    for single_obj_id in obj_ids:
        job_name = f"obj3d_{single_obj_id}"
        script_content = f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_outs/{job_name}-%j.out
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --gpus={gpu}

# Load any necessary modules or activate conda environment here if needed

# Run the command
python run.py --obj_id {single_obj_id} --camera_dist_min {camera_dist_min} --camera_dist_max {camera_dist_max} --scene_scale {scene_scale} --num_images {num_images}
"""
        
        script_path = f'slurm_scripts/{job_name}.sh'
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"Generated Slurm script: {script_path}")
        
        # Submit the job
        if len(obj_ids) == 1:
            submit = input("Submit job to Slurm? (y/n): ").lower().strip()
            if submit == 'y':
                result = subprocess.run(['sbatch', script_path], check=True, capture_output=True, text=True)
                print(f"Job submitted: {result.stdout}")
            else:
                print(f"Job not submitted. You can submit it manually with: sbatch {script_path}")
        else:
            # For multiple jobs, ask once at the beginning
            if single_obj_id == obj_ids[0]:
                submit_all = input(f"Submit all {len(obj_ids)} jobs to Slurm? (y/n): ").lower().strip()
            
            if submit_all == 'y':
                result = subprocess.run(['sbatch', script_path], check=True, capture_output=True, text=True)
                print(f"Job for {single_obj_id} submitted: {result.stdout}")
            else:
                print(f"Job for {single_obj_id} not submitted. You can submit it manually with: sbatch {script_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate and submit a Slurm job for 3D object processing')
    parser.add_argument('--obj_id', type=str, help='Single Objaverse object ID to process')
    parser.add_argument('--obj_ids_file', type=str, help='Path to file containing object IDs, one per line')
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
    
    args = parser.parse_args()
    
    # Determine object IDs to process
    obj_ids = None
    if args.obj_id:
        obj_ids = args.obj_id
    elif args.obj_ids_file:
        with open(args.obj_ids_file, 'r') as f:
            obj_ids = [line.strip() for line in f if line.strip()]
    else:
        parser.error("Either --obj_id or --obj_ids_file must be provided")
    
    generate_slurm_job(
        obj_id=obj_ids,
        camera_dist_min=args.camera_dist_min,
        camera_dist_max=args.camera_dist_max,
        scene_scale=args.scene_scale,
        num_images=args.num_images,
        time=args.time,
        partition=args.partition,
        qos=args.qos,
        gpu=args.gpu,
        mem=args.mem,
        cpus=args.cpus
    ) 