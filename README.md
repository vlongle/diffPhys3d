Our key contribution
- pretrain everything and fast inference
- material field distribution instead of params.


## Install

```
conda create -n diffphys3d python=3.8
conda activate diffphys3d
```
Install torch, torchvision according to your system's cuda version. On my desktop
```
pip install torch torchvision
```
on the grasp cluster,
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then,
```
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install -r requirements.txt
```

Nerfstudio specific stuff:
```
cd third_party/nerfstudio
pip install -e .
cd ../../
cd third_party/f3rm
pip install -e .
## will take forever to install tiny-cuda-nn
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install viser==0.2.7
pip install tyro==0.6.6
```

Install objaverse for blender's python

```
/mnt/kostas-graid/sw/envs/vlongle/blender/blender-4.3.2-linux-x64/4.3/python/bin/python3.11 -m pip install objaverse
```
Might have to install pip for older blender version:
```
# First, download get-pip.py
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

# Then install pip using Blender's Python
/home/vlongle/code/diffPhys3d/objaverse_renderer/blender-3.2.2-linux-x64/3.2/python/bin/python3.10 get-pip.py

# Now you should be able to install objaverse
/home/vlongle/code/diffPhys3d/objaverse_renderer/blender-3.2.2-linux-x64/3.2/python/bin/python3.10 -m pip install objaverse
```

Install some f3rm robot specific stuff
```
pip install --upgrade PyMCubes==0.1.4
pip install params-proto python-slugify
```


We also modify the BlenderNerf add-on to allow random sampling of sphere radius. Under `third_party/BlenderNerf-main-custom`. You should zip this folder and install it as a Blender add-on using the GUI.


Install PhysGaussian dependencies
```
cd third_party/PhysGaussian
pip install -e gaussian-splatting/submodules/simple-knn/
pip install -e gaussian-splatting/submodules/diff-gaussian-rasterization/
```

## Workflow

```
python run.py --obj_id ecb91f433f144a7798724890f0528b23 --camera_dist_min 1.2 --camera_dist_max 1.8 --scene_scale 1.0 --num_images 200
```
This code will:
1. Render nerf data using our custom BlenderNerf add-on.
2. Convert the nerf data to a format that can be used by nerfstudio.
3. Train a f3rm model (nerf + clip distilled feature)
4. Voxelize the scene to obtain the feature grid.


Run physics simulation

```
xvfb-run -a  python gs_simulation_pc.py --point_cloud_path /home/vlongle/code/diffPhys3d/render_outputs/ecb91f433f144a7798724890f0528b23/clip_features_pc.ply --output_path nerf_pc_ununiform_custom_output --config ./config/custom_config.json --render_img --compile_video --white_bg --debug
```

## Command
My Blender version: 4.3.2
### Curating objaverse dataset
```bash
python refine_objaverse_dataset.py
```
to get uids for each object category.

```bash
python curate_objaverse.py
```
to download the objaverse dataset.

```bash
python render_objaverse.py 
```
to render the objaverse dataset.

Data `glb` files are stored in `./assets/` and rendered images are stored in `./render_output/`.

### 
1. Generate data
```bash
cd /home/vlongle/code/diffPhys3d
blender --background --python generate_blendernerf_data.py -- --object_path /home/vlongle/.objaverse/hf-objaverse-v1/glbs/000-064/ecb91f433f144a7798724890f0528b23.glb --num_images 100 --format NERF --camera_dist 1.8
```
set `--format` to `NERF` for GS and `NGP` for Instant-NGP (faster NERF).

2. Train GS
```bash
cd /home/vlongle/code/gaussian-splatting
python train.py -s data/ecb91f433f144a7798724890f0528b23 -w --iterations 10_000
```

3. Train PhysGaussian
```bash
cd /home/vlongle/code/PhysGaussian
python gs_simulation.py --model_path ./model/test_white_bg_prune --output_path custom_output --config ./config/custom_config.json --render_img --compile_video --white_bg --debug
```

Available materials in PhysGaussian:
jelly, metal, sand, foam, snow and plasticine

