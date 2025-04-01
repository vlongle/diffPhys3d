
```
 ns-render dataset --load-config  outputs/ecb91f433f144a7798724890f0528b23/f3rm/2025-03-19_182730/config.yml --output-path test_nerf_renders_outputs --split=train --rendered_output_names=feature_pca
```

```
python extract_clip_voxels.py \
    --scene outputs/ecb91f433f144a7798724890f0528b23/f3rm/2025-03-19_182730/config.yml \
    --output outputs/ecb91f433f144a7798724890f0528b23/f3rm/2025-03-19_182730/feature_grid.npz \
    --voxel_size 0.005
```



```
python extract_clip_voxels.py \
    --scene outputs/ecb91f433f144a7798724890f0528b23/f3rm/2025-03-19_182730/config.yml \
    --output outputs/ecb91f433f144a7798724890f0528b23/f3rm/2025-03-19_182730/feature_grid_voxel_0.01.npz \
    --voxel_size 0.01
```


python extract_clip_voxels.py \
    --scene outputs/ecb91f433f144a7798724890f0528b23/f3rm/2025-03-19_182730/config.yml \
    --output outputs/ecb91f433f144a7798724890f0528b23/f3rm/2025-03-19_182730/feature_grid_voxel_0.1.npz \
    --voxel_size 0.1


```
ns-render dataset --load-config  outputs/ecb91f433f144a7798724890f0528b23/f3rm/2025-03-19_182730/config.yml --output-path test_nerf_renders_outputs --split=train --rendered_output_names=grid_features_pca
```


```
    def populate_modules(self):
```
initialize the feature field prediction



ns-render dataset --load-config  outputs/ecb91f433f144a7798724890f0528b23/f3rm/2025-03-19_182730/config.yml --output-path test_nerf_renders_0.01_outputs --split=train --rendered_output_names=grid_features_pca



python extract_clip_voxels.py \
    --scene outputs/ecb91f433f144a7798724890f0528b23/f3rm/2025-03-19_182730/config.yml \
    --output outputs/ecb91f433f144a7798724890f0528b23/f3rm/2025-03-19_182730/feature_grid_voxel_0.001.npz \
    --voxel_size 0.001



```
ns-viewer --load-config  outputs/ecb91f433f144a7798724890f0528b23/f3rm/2025-03-19_182730/config.yml
```


```
ns-train f3rm --data ficus_transparent_bg/ecb91f433f144a7798724890f0528b23 --max-num-iterations 5000
```


## HACKY:
to fix this: righjt now, in
```
        grid_feature_path: Optional[str] = "outputs/ecb91f433f144a7798724890f0528b23/f3rm/2025-03-19_182730/feature_grid_voxel_0.01.npz",
```
is hardcoded within `feature_field.py`. Which is not a good practice. TODO later is to configurize it somehow.






## Fix nerfstudio
Then follow this https://github.com/nerfstudio-project/nerfstudio/issues/2615
to fix the eager execution error.


https://github.com/pmneila/PyMCubes/issues/49

```
pip install --upgrade PyMCubes==0.1.4
```

Add
```
import torch._dynamo
torch._dynamo.config.suppress_errors = True
```
not only to the train.py of nerfstudio source code but also optimize.py of f3rm_robot.


Must use nerfstudio version 0.3.4.


## Voxelization debugging


Theirs:
```
f3rm-optimize --scene outputs/scene_001/f3rm/2025-03-24_195531/config.yml --visualize false
```
Ours:
```
f3rm-optimize --scene  outputs/ecb91f433f144a7798724890f0528b23/f3rm/2025-03-19_182730/config.yml visualize false
```


But dataset is only available at around 1.4.6 nerfstudio or something. BUGGER.


## BUGS

For some reason, the scene box is a two-cubic cube instead of a unit cube. 
TODO: re-generate data from blender_nerf. Does all the check ect.


```
f3rm-optimize --scene  outputs/ecb91f433f144a7798724890f0528b23/f3rm/2025-03-24_220106/config.yml
```

TODO: Weird bug where the scene bounds are very weird. 



This is the one that we use for rendering because it supports dataset
```
pip show nerfstudio | grep Version
Version: 1.1.5
```


```
conda create -n nerfstudio python=3.8
conda activate nerfstudio
pip install torch torchvision 
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install nerfstudio==0.3.0
pip install viser==0.1.12  
pip install tyro==0.6.6
```

Then follow this https://github.com/nerfstudio-project/nerfstudio/issues/2615
to fix the eager execution error.




```
ns-render dataset --load-config  outputs/ecb91f433f144a7798724890f0528b23/f3rm/2025-03-24_235407/config.yml --output-path clip_grid_outputs --split=train --rendered_output_names=similarity --lang_positives pot --grid_feature_path  outputs/ecb91f433f144a7798724890f0528b23/f3rm/2025-03-24_235407/clip_features.npz
```


```
 python extract_clip_voxels.py --scene  outputs/ecb91f433f144a7798724890f0528b23/f3rm/2025-03-24_235407/config.yml --output  outputs/ecb91f433f144a7798724890f0528b23/f3rm/2025-03-24_235407/clip_features.npz --voxel_size 0.01
```



TODO:
- the feature pca grid still looks incorrect...




ns-render dataset --load-config  outputs/39c14b1d1d63467588ab6bd44a5525d0/f3rm/2025-03-25_223114/config.yml --output-path clip_grid_outputs/39c14b1d1d63467588ab6bd44a5525d0 --split=train --rendered_output_names=similarity --lang_positives pot --grid_feature_path  outputs/ecb91f433f144a7798724890f0528b23/f3rm/2025-03-24_235407/clip_features.npz



ns-render dataset --load-config  outputs/39c14b1d1d63467588ab6bd44a5525d0/f3rm/2025-03-25_223114/config.yml --output-path clip_grid_outputs/39c14b1d1d63467588ab6bd44a5525d0 --split=train --rendered_output_names=feature_pca --lang_positives pot --grid_feature_path  outputs/ecb91f433f144a7798724890f0528b23/f3rm/2025-03-24_235407/clip_features.npz


TODO:
