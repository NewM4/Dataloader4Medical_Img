import SimpleITK as sitk
import pandas as pd
import numpy as np
import radiomics
import radiomics.featureextractor as FEE
import os


def get_img_path(img_path, mask_path):
    img_dir_list = [os.path.join(img_path, temp_dir) for temp_dir in os.listdir(img_path)]
    mask_dir_list = [os.path.join(mask_path, temp_dir) for temp_dir in os.listdir(mask_path)]
    return os.listdir(img_path), img_dir_list, mask_dir_list


def feature_extract_to_csv(ori_path, mask_path, para_path=None, csv_path = "./result.csv"):
    """
    parameters:
        ori_path: 原图像文件
        mask_path: mask图像文件
        para_path: 用于提取特征的配置文件
        csv_path: 要保存的csv的地址，不指定默认为当前文件夹："./result.csv"
    return:
        result_dic: 返回提取出的特征的字典
    """
    extractor = FEE.RadiomicsFeatureExtractor(para_path)
    # print("Extraction parameters: \n\t", extractor.settings)
    # print("Enabled filters: \n\t", extractor.enabledImagetypes)
    # print ("Enabled features:\n\t", extractor.enabledFeatures)
    result = extractor.execute(ori_path, mask_path)  #抽取特征
    result_odict = result.items()
    temp_df = pd.DataFrame(result_odict, columns=['key','value'])
    temp_df.to_csv(csv_path, encoding="utf8", index=None)

    return result_odict


if __name__ == "__main__":
    extractor = FEE.RadiomicsFeatureExtractor()  # use defaults
    img_list, img_dir_list, mask_dir_list = get_img_path("../dataset/ct", "../dataset/label")
    i = 0
    for per_img_path, per_mask_path in zip(img_dir_list, mask_dir_list):
        print(f"The {i}-th img, named: {img_list[i]}, processing...")
        feature_extract_to_csv(per_img_path, per_mask_path, csv_path=f"../dataset/result_{img_list[i]}.csv")
        i += 1
