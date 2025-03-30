from typing import Dict, Optional, Tuple

import numpy as np
import tinycudann as tcnn
from jaxtyping import Float, Shaped
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field
from torch import Tensor
import torch
import os
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.scene_box import SceneBox

class FeatureFieldHeadNames:
    GRID_FEATURES: str = "grid_features"
    FEATURE: str = "feature"


class FeatureField(Field):
    aabb: Tensor

    def __init__(
        self,
        feature_dim: int,
        spatial_distortion: SpatialDistortion,
        aabb: Tensor,
        # Positional encoding
        use_pe: bool = True,
        pe_n_freq: int = 6,
        # Hash grid
        num_levels: int = 12,
        log2_hashmap_size: int = 19,
        start_res: int = 16,
        max_res: int = 128,
        features_per_level: int = 8,
        # MLP head
        hidden_dim: int = 64,
        num_layers: int = 2,
        # Grid features
        grid_feature_path: Optional[str] = None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.spatial_distortion = spatial_distortion
        self.register_buffer("aabb", aabb)
        CONSOLE.print(f">> DEBUG: F3RM FEATURE_FIELD.PY:Spatial distortion: {self.spatial_distortion}")
        CONSOLE.print(f">> DEBUG: F3RM FEATURE_FIELD.PY:AABB: {aabb}")

        # Feature field has its own hash grid
        growth_factor = np.exp((np.log(max_res) - np.log(start_res)) / (num_levels - 1))
        encoding_config = {
            "otype": "Composite",
            "nested": [
                {
                    "otype": "HashGrid",
                    "n_levels": num_levels,
                    "n_features_per_level": features_per_level,
                    "log2_hashmap_size": log2_hashmap_size,
                    "base_resolution": start_res,
                    "per_level_scale": growth_factor,
                }
            ],
        }

        if use_pe:
            encoding_config["nested"].append(
                {
                    "otype": "Frequency",
                    "n_frequencies": pe_n_freq,
                    "n_dims_to_encode": 3,
                }
            )

        self.field = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=self.feature_dim,
            encoding_config=encoding_config,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers,
            },
        )

        # Load grid features
        self.grid_features = None
        self.set_grid_feature(grid_feature_path)



    def set_grid_feature(self, grid_feature_path):
        self.grid_metadata = None
        print(f">> DEBUG: FEATURE_FIELD.PY:Loading grid features from {grid_feature_path}")
        if grid_feature_path and os.path.exists(grid_feature_path):
            try:
                # Load metadata
                metadata = np.load(grid_feature_path)
                
                # Check if features are stored separately
                features_path = grid_feature_path.replace('.npz', '_features.npy')
                if os.path.exists(features_path):
                    features = np.load(features_path)
                    self.grid_features = torch.from_numpy(features).float()
                    print(f"Loaded grid features with shape {self.grid_features.shape}")
                else:
                    print(f"Warning: Could not find features file at {features_path}")
                    
                # Store metadata
                self.grid_metadata = {k: metadata[k] for k in metadata.keys()}
            except Exception as e:
                print(f"Error loading grid features: {e}")

    def get_density(
        self, ray_samples: RaySamples
    ) -> Tuple[Shaped[Tensor, "*batch 1"], Float[Tensor, "*batch num_features"]]:
        raise NotImplementedError("get_density not supported for FeatureField")

    ## TODO: HACK: debug the grid features stuff. Doens't work as expected.
    # def get_grid_features(self, ray_samples: RaySamples) -> Tensor:
    #     ### based on tensoRFField
    #     # d = ray_samples.frustums.directions
    #     # positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
    #     random_positions = torch.rand(ray_samples.frustums.directions.shape[0], 3, device=ray_samples.frustums.directions.device)
    #     return random_positions

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        outputs = {}
        # Apply scene contraction
        ## HACK: New code
        # grid_features = self.get_grid_features(ray_samples)
        # grid_features = self.get_grid_features_nearest(positions)
        # outputs[FeatureFieldHeadNames.GRID_FEATURES] = grid_features


        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions().detach()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)

        positions_flat = positions.view(-1, 3)

        # Get features from the neural network
        features = self.field(positions_flat).view(*ray_samples.frustums.directions.shape[:-1], -1)
        
        outputs[FeatureFieldHeadNames.FEATURE] = features
        outputs[FeatureFieldHeadNames.GRID_FEATURES] = features ## HACK: TODO: debug the grid features issue

        return outputs

    # def get_grid_features(self, positions: Tensor) -> Tensor:
    #     # Add grid features if available
    #     if self.grid_features is not None:
    #         # Get normalized positions for grid sampling
    #         if self.grid_metadata and 'min_bounds' in self.grid_metadata and 'max_bounds' in self.grid_metadata:
    #             min_bounds = torch.tensor(self.grid_metadata['min_bounds'], device=positions.device, dtype=torch.float32)
    #             max_bounds = torch.tensor(self.grid_metadata['max_bounds'], device=positions.device, dtype=torch.float32)
                
    #             # Normalize positions to [0, 1] based on grid bounds
    #             norm_positions = (positions - min_bounds) / (max_bounds - min_bounds)
    #         else:
    #             print(f">> DEBUG: FEATURE_FIELD.PY:No grid metadata found")
    #             # Assume positions are already in [0, 1]
    #             norm_positions = positions
                
    #         # Convert to [-1, 1] for grid_sample
    #         grid_positions = norm_positions * 2.0 - 1.0
            
    #         # Ensure correct data type (float32)
    #         grid_positions = grid_positions.to(torch.float32)
            
    #         # Reshape for grid_sample
    #         original_shape = positions.shape[:-1]
    #         grid_positions_reshaped = grid_positions.reshape(1, 1, 1, -1, 3)
            
    #         # Ensure grid_features is on the same device as positions and has correct dtype
    #         if self.grid_features.device != positions.device or self.grid_features.dtype != torch.float32:
    #             self.grid_features = self.grid_features.to(device=positions.device, dtype=torch.float32)
            
    #         # Reshape grid_features for grid_sample
    #         feature_dim = self.grid_features.shape[-1]
    #         grid_features_reshaped = self.grid_features.permute(3, 0, 1, 2).unsqueeze(0)
            
    #         # Perform trilinear interpolation
    #         sampled_features = torch.nn.functional.grid_sample(
    #             grid_features_reshaped,
    #             grid_positions_reshaped,
    #             mode='bilinear',  # 'bilinear' in 3D is actually trilinear
    #             align_corners=True,
    #             padding_mode='border'
    #         )
            
    #         # Reshape to match the original shape
    #         sampled_features = sampled_features.squeeze(0).squeeze(1).squeeze(1)
    #         sampled_features = sampled_features.permute(1, 0).reshape(*original_shape, feature_dim)

    #         return sampled_features
        
    #     # Return None or a default value if grid_features is not available
    #     return None


    def get_grid_features_nearest(self, positions: Tensor) -> Tensor:
        """
        Get grid features using simple nearest neighbor lookup with vectorized operations.
        
        Args:
            positions: Tensor of shape (..., 3) containing the positions to query
            
        Returns:
            Tensor of shape (..., feature_dim) containing the features at the queried positions,
            or None if grid_features is not available
        """
        if self.grid_features is None:
            print("No grid features available")
            return None
        
        # Get grid dimensions and bounds
        if self.grid_metadata and 'min_bounds' in self.grid_metadata and 'max_bounds' in self.grid_metadata:
            min_bounds = torch.tensor(self.grid_metadata['min_bounds'], device=positions.device, dtype=torch.float32)
            max_bounds = torch.tensor(self.grid_metadata['max_bounds'], device=positions.device, dtype=torch.float32)
        else:
            print("WARNING: No bounds in metadata, using default [-0.5, 0.5] range")
            min_bounds = torch.tensor([-0.5, -0.5, -0.5], device=positions.device, dtype=torch.float32)
            max_bounds = torch.tensor([0.5, 0.5, 0.5], device=positions.device, dtype=torch.float32)
        
        # Get grid dimensions
        grid_shape = self.grid_features.shape[:-1]  # (depth, height, width)
        
        # Calculate the size of each voxel
        voxel_size = (max_bounds - min_bounds) / torch.tensor([grid_shape[2], grid_shape[1], grid_shape[0]], 
                                                             device=positions.device)
        
        # Calculate grid indices directly
        grid_indices = ((positions - min_bounds) / voxel_size).long()
        
        # Clamp indices to valid range
        grid_indices[..., 0] = torch.clamp(grid_indices[..., 0], 0, grid_shape[2] - 1)  # x -> width
        grid_indices[..., 1] = torch.clamp(grid_indices[..., 1], 0, grid_shape[1] - 1)  # y -> height
        grid_indices[..., 2] = torch.clamp(grid_indices[..., 2], 0, grid_shape[0] - 1)  # z -> depth
        
        # Save original shape for reshaping the output
        original_shape = positions.shape[:-1]
        
        # Reshape indices for indexing
        flat_indices = grid_indices.reshape(-1, 3)
        
        # Make sure grid_features and indices are on the same device
        if self.grid_features.device != flat_indices.device:
            self.grid_features = self.grid_features.to(device=flat_indices.device)
        
        # Use advanced indexing to get features (vectorized)
        # Note: grid_indices is [x, y, z] but grid_features is indexed as [z, y, x, features]
        sampled_features = self.grid_features[flat_indices[:, 2], flat_indices[:, 1], flat_indices[:, 0]]
        
        # Reshape back to original shape
        sampled_features = sampled_features.reshape(*original_shape, -1)
        
        return sampled_features

    def forward(self, ray_samples: RaySamples, compute_normals: bool = False) -> Dict[FieldHeadNames, Tensor]:
        if compute_normals:
            raise ValueError("FeatureField does not support computing normals")
        return self.get_outputs(ray_samples)
