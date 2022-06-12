import os
import numpy as np
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset
from utils.transform import *


class MriDataset(Dataset):
    """
    
    """
    def __init__(self, mri_path, mask_path):
        super().__init__()
        self.transforms = Compose([
            # ToTensor(),
            # Resize(),
            RandomCrop(48),
            RandomFlip_Image(),
            RandomFlip_Image(height=False),
            RandomFlip_Slice(),
        ])

        # mri和mask文件路径
        self.mri_path_list = [os.path.join(mri_path, file) for file in os.listdir(mri_path)]
        # 扩充两倍数据
        self.mri_path_list += self.mri_path_list

        self.mask_path_list = [os.path.join(mask_path, file) for file in os.listdir(mask_path)]
        # 扩充两倍数据
        self.mask_path_list += self.mask_path_list


    def __getitem__(self, item):
        # 首先根据路径读入数据，再构造
        mri = sitk.ReadImage(self.mri_path_list[item], sitk.sitkInt16)
        mask = sitk.ReadImage(self.mask_path_list[item], sitk.sitkUInt8)
        # 转换为np.ndarray矩阵
        mri_array = sitk.GetArrayFromImage(mri)
        mask_array = sitk.GetArrayFromImage(mask)
        # 将mask分解为不同类别的array，有几个类别就分解为几个矩阵
        obj_ids = np.unique(mask_array)  # 计算mask中有几个类别
        temp_masks = mask_array == obj_ids[:, None, None, None]  # (n_classes, 135, 256, 256)举个例子
        masks = torch.as_tensor(temp_masks, dtype=torch.float32)
        # is_mask = np.zeros((mask_array.shape[0], ))
        # for i in range(mask_array.shape[0]):
        #     if np.any(mask_array[i] > 0):
        #         is_mask[i] = 1

        # 给矩阵加一个channel，在第0维度上
        mri_array = torch.unsqueeze(torch.tensor(mri_array, dtype=torch.float32), 0)

        if transforms is not None:
            mri_array, masks = self.transforms(mri_array, masks)

        return mri_array, masks

    def __len__(self):
        return len(self.mri_path_list)


if __name__ == "__main__":
    mri_dataset = MriDataset("./train/ct", "./train/label")
    print(len(mri_dataset))
    from torch.utils.data import DataLoader
    dataloader = DataLoader(mri_dataset, batch_size=2)
    next(iter(dataloader))
    for i, (img, masks) in enumerate(dataloader):
        print(f"当前图像：{img.size()}, {masks.size()}")
        break

    print('')
