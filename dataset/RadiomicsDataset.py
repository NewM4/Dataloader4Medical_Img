import torch
import pandas as pd
import torch.nn as nn
import os


class RadiomicsDataset(nn.Module):
    def __init__(self, file_path):
        super().__init__()
        self.file_list = os.listdir(file_path)
        self.file_list += self.file_list
        self.file_path_list = [os.path.join(file_path, per_file_name) for per_file_name in self.file_list]
        self.file_path_list += self.file_path_list


    def __getitem__(self, item):
        temp_df = pd.read_csv(self.file_path_list[item])
        features_array = torch.from_numpy(temp_df["value"].values)
        features_array = features_array.to(torch.float32)
        return features_array

    def __len__(self):
        return len(self.file_path_list)


if __name__ == "__main__":
    rdataset = RadiomicsDataset(file_path="../dataset/sequence_features/processed")
    print(len(rdataset))
    from torch.utils.data import DataLoader
    dataloader = DataLoader(rdataset, batch_size=2)
    next(iter(dataloader))
    temp = rdataset[0]
    print("")