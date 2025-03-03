import multiprocessing
import objaverse
import subprocess
import glob
import random

processes = multiprocessing.cpu_count()
lvis_annotations = objaverse.load_lvis_annotations()
print(lvis_annotations.keys())
print(lvis_annotations["fern"])

# # Define where to store organized files
# where_to_store = "./assets"

# # Maximum number of objects to download per class
# max_objs_per_class = 5  # Change this to your desired limit
# obj_list = ["mug"]


# # For each category you want to download
# for key in obj_list:
#     print(key)
    
#     # Get UIDs for this category
#     category_uids = lvis_annotations[key]
    
#     # Limit the number of UIDs if needed
#     if len(category_uids) > max_objs_per_class:
#         # Randomly sample to get a subset
#         category_uids = random.sample(category_uids, max_objs_per_class)
#         print(f"Limiting {key} to {max_objs_per_class} objects (from {len(lvis_annotations[key])} total)")
#     else:
#         print(f"Downloading all {len(category_uids)} {key} objects")
    
#     # Download the limited set of objects
#     objects = objaverse.load_objects(
#         uids=category_uids,
#         download_processes=processes
#     )
    
#     # Create a directory for this category
#     subprocess.call(['mkdir', '-p', f'{where_to_store}/{key}/'])
    
#     # Move all downloaded files to the category directory
#     for folder in glob.glob('/home/vlongle/.objaverse/hf-objaverse-v1/glbs/*/'):
#         subprocess.call(['mv', folder, f'{where_to_store}/{key}/'])



