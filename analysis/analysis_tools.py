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
import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve,auc
from sklearn.metrics import confusion_matrix
from converter.common_utils import hdf5_reader


def plot_confusion_matrix(cm,labels_name,title,save_path):
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
  plt.figure(figsize=(8,8))
  plt.imshow(cm, interpolation='nearest',cmap=plt.cm.Blues)   
  plt.title(title,fontsize='xx-large')    
  plt.colorbar()
  indices = np.array(range(len(labels_name)))    
  plt.xticks(indices, labels_name, fontsize='xx-large')    
  plt.yticks(indices, labels_name, fontsize='xx-large')                
  plt.ylabel('True label',fontsize='xx-large')    
  plt.xlabel('Predicted label',fontsize='xx-large')
  for first_index in range(len(cm)):    
      for second_index in range(len(cm[first_index])):    
          percent =  cm[first_index][second_index]*100
          color = 'black'if percent < 50 else 'w'
          plt.text(first_index, second_index, '%.2f%%'%(cm[first_index][second_index]*100),color=color,
                  verticalalignment='center',horizontalalignment='center',fontsize='xx-large')
  plt.savefig('{}/{}.png'.format(save_path,title.replace(' ','_')), format='png')
  plt.show()


def plot_roc_curve(true,prob,labels_name,title,save_path):  
  color_list = ['']
  plt.figure(figsize=(8,8))
  lw = 4
  plt.figure(figsize=(10,10))
  for index in range(len(labels_name)):
      fpr,tpr,threshold = roc_curve(y_true=true,y_score=prob[:,index],pos_label=index) 
      roc_auc = auc(fpr,tpr) 
      plt.plot(fpr, tpr, 
                lw=lw, label='%s(AUC = %0.3f)' % (labels_name[index],roc_auc)) 
      plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('1 - Specificity',fontsize='xx-large')
  plt.ylabel('Sensitivity',fontsize='xx-large')
  plt.title(title,fontsize='xx-large')
  plt.legend(loc="lower right",fontsize='xx-large')
  plt.savefig('{}/{}.png'.format(save_path,title.replace(' ','_')), format='png')
  plt.show()


def get_fc_weight(net,weight_path):
  checkpoint = torch.load(weight_path)
  net.load_state_dict(checkpoint['state_dict']) 
  fc_weight = net.fc.weight.detach().numpy()
  # print(type(fc_weight))
  return fc_weight
 
def calculate_CAMs(features, weight, classes_idx):
    """
    Args:
        features (array,(c, h, w)): Output features of last convolution layer.
        weight (array, (classes, c)): weight of linear layer(FC layer).
        classes_idx (list, (classes)): Index of classes.
    Returns:
        cams (list of array, (classes, h, w)): Class Activation Mapping
    """
    c, h, w = features.shape
    cams = []
    for idx in classes_idx:
        cam = weight[idx].dot(features.reshape((c, h*w))) 
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  #Normalize
        cam_img = np.uint8(255 * cam_img) 
        cams.append(cam_img)
    return cams

def save_heatmap(cams, img_path, class_idx, cam_path, transform = None):
    """
    Args:
        cams (array,(c, h, w)): Output features of last convolution layer.
        img_path (str): Path to the original image
        class_idx (int): Index of the class.
        cam_path(str): Path to the heatmap folder
    Returns:
    """
    img_name = img_path.split("/")[-1]
    img = cv2.imread(img_path)
    if transform is not None:
        img = transform(img)
    h, w, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(cams[class_idx], (w, h)), cv2.COLORMAP_JET)  
    result = heatmap * 0.3 + img * 0.5
    
    if not os.path.exists(cam_path):
        os.mkdir(cam_path)
        
    cv2.imwrite(cam_path + img_name + '_' + 'pred_' + str(class_idx) + '.jpg', result) 
    

def my_save_heatmap(cams, img_path, pred_label, true_label, cam_path, transform=None):
    """
    Args:
        cams (array, (c, h, w)): Output features of last convolution layer.
        img_path (str): Path to the original image
        pred_label (int): Predicted class index.
        true_label (int): True class index.
        cam_path (str): Path to the heatmap folder
        transform (callable, optional): A function/transform to apply to the image before saving.
    """
    img_name = img_path.split("/")[-1]
    img = cv2.imread(img_path)
    if transform is not None:
        img = transform(img)
    h, w, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(cams[pred_label], (w, h)), cv2.COLORMAP_JET)  
    result = heatmap * 0.3 + img * 0.5
    
    if not os.path.exists(cam_path):
        os.makedirs(cam_path)

    # Determine if the prediction is correct
    correctness = "Correct" if pred_label == true_label else "Wrong"
    
    # Construct the output filename
    output_filename = f"{img_name}_pred_{pred_label}_true_{true_label}_{correctness}.jpg"
    
    # Save the result image
    cv2.imwrite(os.path.join(cam_path, output_filename), result)