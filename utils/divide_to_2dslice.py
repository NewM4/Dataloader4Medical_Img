import torch
import SimpleITK as sitk
import numpy as np
from torch.utils.data import Dataset
import os


class MriImage(Dataset):
    """
    mri图像数据
    """
    def __init__(self, mri_path, mask_path):
        super().__init__()

        # mri和mask文件路径
        self.mri_path_list = [os.path.join(mri_path, file) for file in os.listdir(mri_path)]
        self.mask_path_list = [os.path.join(mask_path, file) for file in os.listdir(mask_path)]

    def __getitem__(self, item):
        # 首先根据路径读入数据，再构造
        mri = sitk.ReadImage(self.mri_path_list[item], sitk.sitkInt16)
        mask = sitk.ReadImage(self.mask_path_list[item], sitk.sitkUInt8)
        # 转换为np.ndarray矩阵
        mri_array = sitk.GetArrayFromImage(mri)
        mask_array = sitk.GetArrayFromImage(mask)
        # 将mask分解为不同类别的array，有几个类别就分解为几个矩阵
        # obj_ids = np.unique(mask_array)  # 计算mask中有几个类别
        # temp_masks = mask_array == obj_ids[:, None, None, None]  # (n_classes, 135, 256, 256)举个例子
        # masks = torch.as_tensor(temp_masks, dtype=torch.float32)
        is_mask = np.zeros((mask_array.shape[0], ))
        for i in range(mask_array.shape[0]):
            if np.any(mask_array[i] > 0):
                is_mask[i] = 1

        # 给矩阵加一个channel，在第0维度上
        # mri_array = torch.unsqueeze(torch.tensor(mri_array, dtype=torch.float32), 0)

        # target = {
        #     "masks": masks,
        #     "is_mask_vector": is_mask,
        # }

        return mri_array, is_mask

    def __len__(self):
        return len(self.mri_path_list)


def save_per_slices(img, label, img_name, label_name):
    """
    保存每个样本中的单张slice
    :param img: mri图像
    :param label: 每个slice是否有病灶
    :param img_name: 图像的名字
    :return:
    """
    path_mri2d = "../dataset/mri2dslices/mri2d"

    if not os.path.exists(path_mri2d):
        os.mkdir(path_mri2d)
    for i in range(img.shape[0]):
        np.save(os.path.join(path_mri2d, f"{img_name}_{i}.npy"), img[i])


if __name__ == "__main__":
    # 读入数据
    mi_dataset = MriImage(mri_path="../dataset/ct", mask_path="../dataset/label")
    img_name = os.listdir("../dataset/ct")
    label_name = os.listdir("../dataset/label")

    from torch.utils.data import DataLoader
    dataloader = DataLoader(mi_dataset, batch_size=1)
    label_all = []
    for i, (img, label) in enumerate(dataloader):
        # 将每个样本的切片保存
        print(f"处理第{i+1}个...")
        img = img.numpy()[0]
        label_all.append(label.numpy()[0])
        save_per_slices(img, label, img_name[i][:-4], label_name[i][:-4])

    np.save(os.path.join("../dataset/mri2dslices/label", "mri_label.npy"), np.concatenate(label_all))
    print("")
