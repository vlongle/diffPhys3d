run theirs
```
 python gs_simulation.py --model_path ./model/ficus_whitebg-trained --output_path default_output --config ./config/ficus_config.json --render_img --compile_video --white_bg --debug
```

run ours

```
 python gs_simulation.py --model_path ./model/test_white_bg_prune --output_path custom_output --config ./config/custom_config.json --render_img --compile_video --white_bg --debug
 ```


 point_cloud
 ```
 python gs_simulation.py --point_cloud_path model/test_white_bg_prune/point_cloud/iteration_10000/point_cloud.ply  --output_path pc_custom_output --config ./config/custom_config.json --render_img --compile_video --white_bg --debug
 ```



 ## Debuggint the pc stuff

From using the gaussian splatting as point cloud:

```
  python gs_simulation_pc.py --point_cloud_path model/test_white_bg_prune/point_cloud/iteration_10000/point_cloud.ply  --output_path pc_custom_output --config ./config/custom_config.json --render_img --compile_video --white_bg --debug 
```


From voxelizing the groundtruth mesh:

```
  python gs_simulation_pc.py --point_cloud_path ficus_voxel_cache.ply --output_path pc_custom_output --config ./config/custom_config.json --render_img --compile_video --white_bg --debug 
```


From using the nerf voxelized point cloud:

```
 xvfb-run -a  python gs_simulation_pc.py --point_cloud_path /home/vlongle/code/diffPhys3d/render_outputs/ecb91f433f144a7798724890f0528b23/clip_features_pc.ply --output_path nerf_pc_custom_output --config ./config/custom_config.json --render_img --compile_video --white_bg --debug
```



