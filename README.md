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
train feature splatting
```bash
cd /home/vlongle/code/feature-splatting-inria
conda activate feature_splatting

 python compute_obj_part_feature.py -s data/ecb91f433f144a7798724890f0528b23

python train.py -s data/ecb91f433f144a7798724890f0528b23 -m output/ecb91f433f144a7798724890f0528b23 --iterations 10000

python render.py -m output/ecb91f433f144a7798724890f0528b23 -s data/ecb91f433f144a7798724890f0528b23  --camera_slerp_list 0 1 --with_feat --clip_feat --text_query 'a tree pot' --step_size 10

python pca_feature_viz.py --input_dir output/ecb91f433f144a7798724890f0528b23/interpolating_camera/ours_10000/renders --output_dir output/ecb91f433f144a7798724890f0528b23/interpolating_camera/ours_10000/pca_renders

```


View gaussian
```
./home/vlongle/code/gaussian-splatting/SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m  {output_dir}
```

3. Simulate PhysGaussian
```bash
cd /home/vlongle/code/PhysGaussian
python gs_simulation.py --model_path ./model/test_white_bg_prune --output_path custom_output --config ./config/custom_config.json --render_img --compile_video --white_bg --debug
```

Available materials in PhysGaussian:
jelly, metal, sand, foam, snow and plasticine
