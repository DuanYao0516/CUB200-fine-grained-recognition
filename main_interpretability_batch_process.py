'''
Copyright (c) [2024] [Duan Yao in SYSU]

This file is part of the CUB200-fine-grained-recognition project.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:
    http://www.apache.org/licenses/LICENSE-2.0
You must give appropriate credit, provide a link to the license, 
and indicate if changes were made. For any part of your project 
derived from this code, you must explicitly indicate the source.

'''

import os
import csv
import numpy as np
import pandas as pd
import data_utils.transform as tr
from config import INIT_TRAINER
from torchvision import transforms
from converter.common_utils import hdf5_reader
from analysis.analysis_tools import calculate_CAMs, my_save_heatmap
from tqdm import tqdm

            
def find_folder_with_string(root_folder, target_string):
    for root, dirs, files in os.walk(root_folder):  
        for dir in dirs:
            if target_string.lower() in dir.lower():  # 千万注意这一句，因为模型输出mid_feature时把名字改了大小写
                folder_path = os.path.join(root, dir)
                # print("Found folder with target string at: {}".format(folder_path))
                return folder_path

def load_labels_from_csv(csv_path):
    # 读取CSV文件并将其转换为字典，以图像路径为键，真实标签和预测标签为值。
    df = pd.read_csv(csv_path)
    label_dict = {}
    for _, row in df.iterrows():
        label_dict[row['path']] = {'true': row['true'], 'pred': row['pred']}
    return label_dict
            
            
def get_image_paths(features_dir):
    # 获取features_dir文件夹下所有文件的路径
    # 指定数据集路径，也就是原图路径
    img_fold_path = f'./datasets/CUB_200_2011/CUB_200_2011/images/'
    feature_files = [f for f in os.listdir(features_dir) if os.path.isfile(os.path.join(features_dir, f))]
    img_paths = []
    
    for file_name in tqdm(feature_files, total=len(feature_files), desc="Get Original Pics"):
        try:
            # 原数据集中，文件夹名字与mid_feature中的名字不一样，这个问题在 find_folder_with_string 中解决
            class_name = "_".join(file_name.split('_')[:-2])
            img_dir = find_folder_with_string(img_fold_path, class_name)
            img_path = os.path.join(img_dir, file_name + '.jpg')
            img_paths.append(img_path)
        except:
            print(f"Error processing image path for {file_name}. Skipping...")
            print(img_dir,file_name,class_name)
            # 在出现错误时，删除对应的特征文件路径
            feature_files.remove(file_name)  # 删除对应的特征文件路径
    
    return feature_files, img_paths


if __name__ == "__main__":
    # 获取中间特征、模型权重和转换器
    features_dir = './analysis/mid_feature/v1.0/fold1/'  # 需要手工指定
    weight = np.load('./analysis/result/v1.0/fold1_fc_weight.npy')  # 需要手工指定

    transformer = transforms.Compose([
        tr.ToCVImage(),
        tr.RandomResizedCrop(size=INIT_TRAINER['image_size'], scale=(1.0, 1.0)),
        tr.ToTensor(),
        tr.Normalize(INIT_TRAINER['train_mean'], INIT_TRAINER['train_std']),
        tr.ToArray(),
    ])

    # 获取所有 features 和 img 的路径,每一个feature文件对应一个路径
    feature_files, img_paths = get_image_paths(features_dir)

    csv_path = './analysis/result/v1.0/fold1.csv'  # 需要手工指定
    label_dict = load_labels_from_csv(csv_path)

    # 设置类别数与输出路径
    classes = 200
    cam_path = './analysis/result/v1.0/'  # 权重文件，需要自己指定，同时这个路径还是生成热力图路径

    for feature_file, img_path in tqdm(zip(feature_files, img_paths), total=len(feature_files), desc="Generating CAMs"):
        features = hdf5_reader(os.path.join(features_dir, feature_file), 'feature_in')
        # img_name = os.path.basename(img_path)
        if img_path in label_dict:
            true_label = label_dict[img_path]['true']
            pred_label = label_dict[img_path]['pred']
            cams = calculate_CAMs(features, weight, range(classes))
            my_save_heatmap(cams, img_path, pred_label, true_label, cam_path, transform=transformer)
        else:
            print(f"Labels not found for {img_path}. Skipping...")