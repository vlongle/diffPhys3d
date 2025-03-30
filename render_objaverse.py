import os
import argparse
import glob
import os.path as osp
from utils import on_desktop

parser = argparse.ArgumentParser(description='Renders glbs')
parser.add_argument(
    '--save_folder', type=str, default='class_render_outputs',
    help='path for saving rendered image')
parser.add_argument(
    '--folder_assets', type=str, default='./assets/',
    help='path to downloaded 3d assets')
parser.add_argument(
    '--blender_root', type=str, default='blender',
    help='path to blender executable')
parser.add_argument(
    '--obj_class', type=str, default=None,
    help='object class to render (e.g., "mug"). If not provided, renders all available categories')
opt = parser.parse_args()

is_on_desktop = on_desktop()
path_prefix = "/mnt/kostas-graid/datasets/vlongle/diffphys3d" if not is_on_desktop else "."

opt.folder_assets = path_prefix + "/assets"
opt.save_folder = path_prefix + "/class_render_outputs"

# Determine which folders to process based on obj_class
if opt.obj_class:
    # If obj_class is provided, use that specific folder
    asset_path = osp.join(opt.folder_assets, opt.obj_class)
    save_path = osp.join(opt.save_folder, opt.obj_class)
    data = sorted(glob.glob(f"{asset_path}/*/"))
else:
    # If no obj_class is provided, get all category folders
    categories = [osp.basename(f.rstrip('/')) for f in sorted(glob.glob(f"{opt.folder_assets}/*/"))]
    data = []
    for category in categories:
        category_path = osp.join(opt.folder_assets, category)
        data.extend(sorted(glob.glob(f"{category_path}/*/")))
    save_path = opt.save_folder

# Create the base output directory
os.makedirs(save_path, exist_ok=True)

for path in data:
    # path = data[-5]
    path = sorted(glob.glob(path + "/*.glb"))[0]
    
    # Extract a unique identifier from the path (using the directory name)
    dir_name = osp.basename(osp.dirname(path))
    category_name = osp.basename(osp.dirname(osp.dirname(path)))
    
    # Create output path that preserves category structure
    if opt.obj_class:
        output_path = osp.join(save_path, dir_name)
    else:
        output_path = osp.join(save_path, category_name, dir_name)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    render_cmd = '%s -b -P render_blender.py -- --obj %s --output %s --views 1 --resolution 400 > tmp.out' % (
        opt.blender_root, path, output_path
    )
    print(render_cmd)
    os.system(render_cmd)