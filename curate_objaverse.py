import multiprocessing
import objaverse
import subprocess
import glob
import random
import pickle
import os

# Set the number of processes for downloading
processes = multiprocessing.cpu_count()

# Define where to store organized files
where_to_store = "./assets"

# Maximum number of objects to download per class
max_objs_per_class = 5  # just get 5 for testing

# Load the final_dataset.pkl file instead of LVIS annotations
with open("final_dataset.pkl", "rb") as f:
    final_dataset = pickle.load(f)

print("Available categories:", list(final_dataset.keys()))

# Specify which categories you want to download
# categories_to_download = ["rigid_containers"]  # Change this to your desired categories
categories_to_download = list(final_dataset.keys())

DEFAULT_OBJAVERSE_PATH = "/home/vlongle/.objaverse/hf-objaverse-v1/glbs"
## remove the folder before populating
os.system(f"rm -rf {where_to_store}")
os.system(f"mkdir -p {DEFAULT_OBJAVERSE_PATH}")


# For each category you want to download
for category in categories_to_download:
    print(f"Processing category: {category}")
    
    # Get UIDs and scores for this category
    category_items = final_dataset.get(category, [])
    
    if not category_items:
        print(f"No objects found for category: {category}")
        continue
    
    # Sort by similarity score (highest first)
    category_items.sort(key=lambda x: x[1], reverse=True)
    
    # Limit the number of items if needed
    if len(category_items) > max_objs_per_class:
        # Take the top max_objs_per_class items with highest similarity scores
        category_items = category_items[:max_objs_per_class]
        print(f"Taking top {max_objs_per_class} {category} objects with highest similarity scores (from {len(final_dataset[category])} total)")
    else:
        print(f"Downloading all {len(category_items)} {category} objects")
    
    # Extract just the UIDs for downloading
    category_uids = [item[0] for item in category_items]
    
    # Download the objects
    objects = objaverse.load_objects(
        uids=category_uids,
        download_processes=processes
    )
    
    # Create a directory for this category
    os.makedirs(f'{where_to_store}/{category}/', exist_ok=True)
    
    # Move all downloaded files to the category directory
    for folder in glob.glob(f'{DEFAULT_OBJAVERSE_PATH}/*/'):
        subprocess.call(['mv', folder, f'{where_to_store}/{category}/'])



