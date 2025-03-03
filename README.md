Our key contribution
- pretrain everything and fast inference
- material field distribution instead of params.


## Command

1. Generate data
```bash
cd /home/vlongle/code/diffPhys3d
blender --background --python generate_blendernerf_data.py -- --object_path /home/vlongle/.objaverse/hf-objaverse-v1/glbs/000-064/ecb91f433f144a7798724890f0528b23.glb --num_images 100 --format NGP --camera_dist 1.8
```

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
