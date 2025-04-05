import torch

class MaterialVoxelDataset(torch.utils.data.Dataset):
    def __init__(self, data_files, feature_files):
        self.data_files = data_files
        self.feature_files = feature_files

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        material_grid = torch.load(self.data_files[idx])
        feature_grid = torch.load(self.feature_files[idx])
        return material_grid, feature_grid
