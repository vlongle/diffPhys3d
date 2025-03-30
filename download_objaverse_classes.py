import multiprocessing
import objaverse
import subprocess
import glob
import pickle
import os
import argparse
from utils import on_desktop

# Set up argument parser
parser = argparse.ArgumentParser(description='Download Objaverse objects by category')
parser.add_argument(
    '--save_folder', type=str, default='assets',
    help='path for saving downloaded 3D assets')
parser.add_argument(
    '--max_objs_per_class', type=int, default=None,
    help='maximum number of objects to download per class. If not specified, downloads all objects.')
parser.add_argument(
    '--obj_class', type=str, default=None,
    help='specific object class to download (e.g., "mug"). If not provided, downloads all categories')
opt = parser.parse_args()

# Set the number of processes for downloading
processes = multiprocessing.cpu_count()

# Apply path prefix based on desktop status
is_on_desktop = on_desktop()
path_prefix = "/mnt/kostas-graid/datasets/vlongle/diffphys3d" if not is_on_desktop else "."

# Define where to store organized files
where_to_store = f"{path_prefix}/{opt.save_folder}"

# Maximum number of objects to download per class
max_objs_per_class = opt.max_objs_per_class

# Load the final_dataset.pkl file
with open("final_dataset.pkl", "rb") as f:
    final_dataset = pickle.load(f)

print("Available categories:", list(final_dataset.keys()))

# Determine which categories to download
if opt.obj_class:
    if opt.obj_class in final_dataset:
        categories_to_download = [opt.obj_class]
    else:
        print(f"Category '{opt.obj_class}' not found in dataset. Available categories: {list(final_dataset.keys())}")
        exit(1)
else:
    categories_to_download = list(final_dataset.keys())

DEFAULT_OBJAVERSE_PATH = "/home/vlongle/.objaverse/hf-objaverse-v1/glbs"
## remove the folder before populating
os.system(f"rm -rf {where_to_store}")
os.system(f"mkdir -p {DEFAULT_OBJAVERSE_PATH}")

# For each category you want to download
for category in categories_to_download:
    print(f"Processing category: {category}")
    
    # Get UIDs for this category
    category_uids = final_dataset.get(category, [])
    
    if not category_uids:
        print(f"No objects found for category: {category}")
        continue
    
    # Limit the number of UIDs if needed
    if max_objs_per_class is not None and len(category_uids) > max_objs_per_class:
        # Take the first max_objs_per_class items deterministically
        category_uids = category_uids[:max_objs_per_class]
        print(f"Limiting {category} to {max_objs_per_class} objects (from {len(final_dataset[category])} total)")
    else:
        print(f"Downloading all {len(category_uids)} {category} objects")
    
    # # Download the limited set of objects
    # objects = objaverse.load_objects(
    #     uids=category_uids,
    #     download_processes=min(processes, len(category_uids)),
    # )
    
    # Create a directory for this category
    os.makedirs(f'{where_to_store}/{category}/', exist_ok=True)
    
    # Move all downloaded files to the category directory
    for folder in glob.glob(f'{DEFAULT_OBJAVERSE_PATH}/*/'):
        subprocess.call(['mv', folder, f'{where_to_store}/{category}/'])



