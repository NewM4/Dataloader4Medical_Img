import pandas as pd
import numpy as np
import os


def process_csv(csv_path, save_path="./result.csv", save_flag=True):
    """
    删除前几行不需要的，将每一个子项包含元组的拆开来，如果有非数值的，量化。
    保存csv文件
    :param csv_path: 将要处理的csv的路径
    :param result_path: 保存路径
    :return: None
    """
    data_df = pd.read_csv(csv_path)
    # 删除前几行
    data_df = data_df[22:]
    if save_flag:
        data_df.to_csv(save_path, encoding="utf8", index=None)


if __name__ == "__main__":
    file_list = os.listdir("../dataset/sequence_features")
    file_list = file_list[1:]
    file_path_list = [os.path.join("../dataset/sequence_features", per_file) for per_file in file_list]
    for i, file_path in enumerate(file_path_list):
        # processing
        process_csv(file_path, save_path=f"../dataset/sequence_features/processed/processed_{file_list[i]}")

