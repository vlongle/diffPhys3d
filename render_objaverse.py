import os
import argparse
import glob
import os.path as osp

parser = argparse.ArgumentParser(description='Renders glbs')
parser.add_argument(
    '--save_folder', type=str, default='render_output/mug',
    help='path for saving rendered image')
parser.add_argument(
    '--folder_assets', type=str, default='./assets/mug/',
    help='path to downloaded 3d assets')
parser.add_argument(
    '--blender_root', type=str, default='blender',
    help='path to blender executable')
opt = parser.parse_args()

# get all the file
data = sorted(glob.glob(f"{opt.folder_assets}/*/"))

for path in data:
    # path = data[-5]
    path = sorted(glob.glob(path + "/*.glb"))[0]
    
    # Extract a unique identifier from the path (using the directory name)
    dir_name = osp.basename(osp.dirname(path))
    output_path = osp.join(opt.save_folder, dir_name)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    render_cmd = '%s -b -P render_blender.py -- --obj %s --output %s --views 1 --resolution 400 > tmp.out' % (
        opt.blender_root, path, output_path
    )
    print(render_cmd)
    os.system(render_cmd)