Our key contribution
- pretrain everything and fast inference
- material field distribution instead of params.


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
