import os
import argparse
import subprocess
import json

def generate_slurm_job_array(obj_ids, obj_paths=None, camera_dist_min=1.2, camera_dist_max=1.8, 
                            scene_scale=1.0, num_images=200, time="24:00:00", 
                            partition="batch", qos="normal", gpu="1", mem="32G", cpus=8,
                            submit=False, array_limit=100):
    """Generate and submit a Slurm job array to run the 3D object processing pipeline."""
    
    os.makedirs('slurm_scripts', exist_ok=True)
    os.makedirs('slurm_outs', exist_ok=True)
    os.makedirs('slurm_data', exist_ok=True)
    
    # Create a data file with all object IDs and paths
    data_file = 'slurm_data/job_array_data.json'
    data = []
    
    for i, obj_id in enumerate(obj_ids):
        obj_path = obj_paths[i] if obj_paths and i < len(obj_paths) else None
        data.append({
            'obj_id': obj_id,
            'obj_path': obj_path
        })
    
    with open(data_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Create a job array script
    job_name = "obj3d_array"
    
    script_content = f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_outs/{job_name}-%A_%a.out
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --gpus={gpu}
#SBATCH --array=0-{len(data)-1}%{array_limit}

# Load any necessary modules or activate conda environment here if needed

# Get the object ID and path from the data file
TASK_ID=$SLURM_ARRAY_TASK_ID
DATA_FILE=slurm_data/job_array_data.json

# Extract the object ID and path for this task
OBJ_ID=$(python -c "import json; data=json.load(open('$DATA_FILE')); print(data[$TASK_ID]['obj_id'])")
OBJ_PATH=$(python -c "import json; data=json.load(open('$DATA_FILE')); path=data[$TASK_ID]['obj_path']; print(path if path else '')")

# Construct the command
if [ -n "$OBJ_PATH" ]; then
    CMD="python run.py --obj_id $OBJ_ID --obj_path $OBJ_PATH --camera_dist_min {camera_dist_min} --camera_dist_max {camera_dist_max} --scene_scale {scene_scale} --num_images {num_images}"
else
    CMD="python run.py --obj_id $OBJ_ID --camera_dist_min {camera_dist_min} --camera_dist_max {camera_dist_max} --scene_scale {scene_scale} --num_images {num_images}"
fi

# Run the command
echo "Running: $CMD"
$CMD
"""
    
    script_path = f'slurm_scripts/{job_name}.sh'
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"Generated Slurm job array script: {script_path}")
    print(f"Job array will process {len(data)} objects with a maximum of {array_limit} running simultaneously")
    
    # Submit the job if requested
    if submit:
        result = subprocess.run(['sbatch', script_path], check=True, capture_output=True, text=True)
        print(f"Job array submitted: {result.stdout}")
        return result.stdout
    else:
        print(f"Job array not submitted. You can submit it manually with: sbatch {script_path}")
        return script_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate and submit a Slurm job array for 3D object processing')
    parser.add_argument('--obj_ids_file', type=str, required=True, help='Path to file containing object IDs, one per line')
    parser.add_argument('--obj_paths_file', type=str, help='Path to file containing object paths, one per line (optional)')
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
    parser.add_argument('--submit', action='store_true', help='Submit the job to Slurm immediately')
    parser.add_argument('--array_limit', type=int, default=100, help='Maximum number of array jobs to run simultaneously')
    
    args = parser.parse_args()
    
    # Read object IDs from file
    with open(args.obj_ids_file, 'r') as f:
        obj_ids = [line.strip() for line in f if line.strip()]
    
    # Read object paths from file if provided
    obj_paths = None
    if args.obj_paths_file:
        with open(args.obj_paths_file, 'r') as f:
            obj_paths = [line.strip() for line in f if line.strip()]
    
    generate_slurm_job_array(
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
        submit=args.submit,
        array_limit=args.array_limit
    ) 