# Dataloader4Medical_Img
> 医学图像的Dataset

# MriDataset
返回病人的mri序列和mask
mri: (batch_size, 1, 48, 512, 512)
mask: (batch_size, 1, 48, 512, 512)<br>
上述"48"经过预处理得到,具体参考文件为: [3DUNet-Pytorch](https://github.com/lee-zq/3DUNet-Pytorch),其中的"preprocess_LiTS.py"

# MriDataset2d
返回mri序列图像的单张slice
mri: (batch_size, 1, 512, 512)
mask: (batch_size, 1, 512, 512)<br>
将mri序列图像处理为单张slice的文件在: "utils/divide_to_2dslice.py"

# RadiomicsDataset
返回在mri序列图像上提取到的影像组学特征<br>
具体的提取文件在utils下.