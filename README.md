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

Install some f3rm robot specific stuff
```
pip install --upgrade PyMCubes==0.1.4
pip install params-proto python-slugify
```
## Workflow
1. Download the object
```
python download_objaverse.py --obj_id ecb91f433f144a7798724890f0528b23
```
2. Render via blender
```
export PATH="/mnt/kostas-graid/sw/envs/vlongle/blender/blender-4.3.2-linux-x64:$PATH";
blender --background --python generate_blendernerf_data.py -- --obj_id ecb91f433f144a7798724890f0528b23 --num_images 100 --format NGP --camera_dist 1.8 --output_dir /mnt/kostas-graid/datasets/vlongle/diffphys3d/data
```
3. Train distilled nerf using f3rm library
Convert data
```
python convert.py --obj_id ecb91f433f144a7798724890f0528b23
```
Train the model
```
ns-train f3rm --data /mnt/kostas-graid/datasets/vlongle/diffphys3d/data/f420ea9edb914e1b9b7adebbacecc7d8 --max-num-iterations 5000
```

Check the results
```
ns-render dataset --load-config outputs/f420ea9edb914e1b9b7adebbacecc7d8/f3rm/2025-03-26_052550/config.yml --output-path clip_grid_outputs/f420ea9edb914e1b9b7adebbacecc7d8 --split=train --rendered_output_names=rgb
```

4. Extract voxels

```
python voxel_to_pc.py --scene outputs/ecb91f433f144a7798724890f0528b23/f3rm/2025-03-26_083416/config.yml --output outputs/ecb91f433f144a7798724890f0528b23/f3rm/2025-03-26_083416/clip_features.npz
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



## Scratch

blender --background --python generate_blendernerf_data.py -- --object_path /home/vlongle/.objaverse/hf-objaverse-v1/glbs/000-064/ecb91f433f144a7798724890f0528b23.glb --num_images 100 --format NGP --camera_dist 1.8 --output_dir /mnt/kostas-graid/datasets/vlongle/diffphys3d/data


ns-train f3rm --data /mnt/kostas-graid/datasets/vlongle/diffphys3d/data/ecb91f433f144a7798724890f0528b23 --max-num-iterations 5000



ns-render dataset --load-config outputs/ecb91f433f144a7798724890f0528b23/f3rm/2025-03-26_060324/config.yml --output-path clip_grid_outputs/ecb91f433f144a7798724890f0528b23 --split=train --rendered_output_names=rgb


## Debugging tips

f3rm_robot/args.py is important for checking min_bounds and max_bounds.





