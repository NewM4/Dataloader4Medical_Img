import os
import numpy as np
from torch.utils.data import Dataset
from utils.transform import *


class MriDataset2D(Dataset):
    """
    构造样本标签对 (instance-label pairs) 为：
        样本：MRI序列图中的单张slice
        标签：0表示当前样本为正常样本，1表示当前样本为病灶样本
    """
    def __init__(self, mri_path, label_path):
        """
        :param mri_path: 单张序列图所在文件夹
        :param label_path: 标签的路径
        """
        super().__init__()
        # self.transforms = Compose([
        #     # ToTensor(),
        #     # Resize(),
        #     RandomCrop(48),
        #     RandomFlip_Image(),
        #     RandomFlip_Image(height=False),
        #     RandomFlip_Slice(),
        # ])

        # mri和mask文件路径
        self.mri_path_list = [os.path.join(mri_path, file) for file in os.listdir(mri_path)]
        self.mri_label = np.load(label_path)

    def __getitem__(self, item):
        # 首先根据路径读入数据，再构造
        mri = np.load(self.mri_path_list[item])
        mri = mri.reshape((1, mri.shape[0], mri.shape[1]))
        label = self.mri_label[item]

        # if transforms is not None:
        #     mri_array, masks = self.transforms(mri)

        return mri, label

    def __len__(self):
        return len(self.mri_path_list)


if __name__ == "__main__":
    mri2d_dataset = MriDataset2D(mri_path="mri2dslices/mri2d", label_path="mri2dslices/label/mri_label.npy")
    from torch.utils.data import DataLoader
    m2_dataloader = DataLoader(mri2d_dataset, batch_size=2, shuffle=False)
    a = next(iter(m2_dataloader))
    pass
