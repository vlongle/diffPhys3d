import torch
import os
import numpy as np
class MaterialVoxelDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.data_files, self.feature_files = self.get_data_files()

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        material_grid = torch.from_numpy(np.load(self.data_files[idx]))
        feature_grid = torch.from_numpy(np.load(self.feature_files[idx]))
        return material_grid, feature_grid


    def get_data_files(self):
        data_files = []
        feature_files = []
        for obj_id in os.listdir(self.data_dir):
            feature_file = os.path.join(self.data_dir, obj_id, "clip_features_features.npy")
            material_file = os.path.join(self.data_dir, obj_id, "material_grid.npy")
            if os.path.exists(feature_file) and os.path.exists(material_file):
                data_files.append(material_file)
                feature_files.append(feature_file)
        return data_files, feature_files


if __name__ == "__main__":
    data_dir = "/mnt/kostas-graid/datasets/vlongle/diffphys3d/render_outputs"
    dataset = MaterialVoxelDataset(data_dir)
    print("len(dataset): ", len(dataset))
    material_grid, feature_grid = dataset[0]
    print("material_grid.shape: ", material_grid.shape)
    print("feature_grid.shape: ", feature_grid.shape)

    ## count how many vector in faeture_grid that have all zeros
    zero_feature_count = (feature_grid == 0).sum()
    print(f"Number of zero vectors in feature_grid: {zero_feature_count}")
    
    ### material grid shape is NxNxNx4, where 4 is [density, E, nu, material_id]
    ## count how many material_id == 7
    material_id_count = (material_grid[:, :, :, 3] == 7).sum()
    print(f"Number of material_id == 7 in material_grid: {material_id_count}")

    assert material_id_count == zero_feature_count, "material_id_count and zero_feature_count should be the same"
