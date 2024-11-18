# main_robustness.py

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

import torch
from trainer import VolumeClassifier  # 确保导入路径正确
from data_utils.csv_reader import csv_reader_single
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


def validate_label_dict(label_dict, num_classes):
    for key, value in label_dict.items():
        if value < 0 or value >= num_classes:
            raise ValueError(f"Invalid label {value} for key {key}. Should be in range [0, {num_classes-1}].")


# result = {'true': [...], 'pred': [...], 'prob': [...]}
def evaluate_results(result):
    true_labels = result['true']
    pred_labels = result['pred']
    probabilities = result['prob']

    # 计算评估指标
    accuracy = accuracy_score(true_labels, pred_labels)  # 计算整体准确率，(TP+FP)/total
    precision = precision_score(true_labels, pred_labels, average='weighted')  # 计算精度， TP/(TP+FP)
    recall = recall_score(true_labels, pred_labels, average='weighted')  # 计算召回率 TP/(TP+FN)
    f1 = f1_score(true_labels, pred_labels, average='weighted')  # 计算F1 分数

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # 生成混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_dict.values(), yticklabels=label_dict.values())
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')  # 保存混淆矩阵图像
    plt.show()  # 混淆矩阵热力图
    

    # # 保存结果到CSV文件
    # result_df = pd.DataFrame({'true': true_labels, 'pred': pred_labels, 'prob': probabilities})
    # result_df.to_csv('attack_results.csv', index=False)
    
    # 保存结果到CSV文件
    true_prob = [prob[true] for true, prob in zip(true_labels, probabilities)]
    pred_prob = [prob[pred] for pred, prob in zip(pred_labels, probabilities)]
    result_df = pd.DataFrame({'true': true_labels, 'pred': pred_labels, 'true_prob': true_prob, 'pred_prob': pred_prob})
    result_df.to_csv('attack_results.csv', index=False)

    # 计算并绘制ROC曲线和AUC
    fpr, tpr, _ = roc_curve(true_labels, [prob[1] for prob in probabilities], pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('ROC.png')  # 保存ROC图像
    plt.show()

    return accuracy, precision, recall, f1

if __name__ == "__main__":
    classifier = VolumeClassifier(
        net_name='resnet50', 
        lr=0.001, 
        n_epoch=1, 
        num_classes=200, 
        image_size=224, 
        batch_size=6, 
        train_mean=[0.485, 0.456, 0.406], 
        train_std=[0.229, 0.224, 0.225], 
        device='0',
        pre_trained=True, 
        weight_path='./ckpt/v6.0/fold1/epoch=3-train_loss=0.68016-val_loss=1.94169-train_acc=0.80241-val_acc=0.51586.pth'  # 预训练为 True 时这一项才有效
        # 更改为最后一次训练时的权重
    )
    
    # test_path = 'path_to_test_data'
    test_csv_path = './csv_file/cub_200_2011.csv_test.csv'
    label_dict = csv_reader_single(test_csv_path, key_col='id', value_col='label')
    # validate_label_dict(label_dict, 200)  # 验证标签
    test_path = list(label_dict.keys())

    result = classifier.inference_with_attack(  # 运行推理并应用对抗攻击
        test_path=test_path, 
        label_dict=label_dict, 
        attack_type='PGD',  # 选择 'FGSM' 或 'PGD'
        epsilon=0.3,   # 以下三个是攻击参数
        alpha=0.01, 
        num_iter=40,
        save_dir='./attack/PGD'  # 新增保存路径参数
    )
    
    # 进一步处理 result，保存结果或计算其他指标 等等等
    
    accuracy, precision, recall, f1 = evaluate_results(result)
    
