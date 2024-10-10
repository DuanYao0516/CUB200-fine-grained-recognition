## CLIP zero-shot
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch
from data_utils.csv_reader import csv_reader_single
from torch.cuda.amp import autocast, GradScaler
import data_utils.transform as tr
from data_utils.data_loader import DataGenerator
from torch.utils.data import DataLoader
from torch.nn import functional as F
import cv2
import numpy as np
# from main import get_cross_validation
# from  trainer import accuracy,AverageMeter
## install packages: torch, transformers==4.23.1

model_path="./model/b16"
para_save_path="./ckpt/"
para_version='v5.0'            #保存参数时的路径
use_para_version='v1.0'        #加载参数时的路径
lr=5e-6
Batch_size=128
Epoch=10
train_load=True

class VLMInterface:
    def __init__(self,load_pretrained=False, config_name=model_path, model_name=model_path):
        
        print('Loading Model, wait for a minute.')
        
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CLIPModel.from_pretrained(model_name)
        
        if load_pretrained:
            print('Loading pre_train_para .')
            self.model.load_state_dict(torch.load(para_save_path+use_para_version, map_location=self.device))
            
        self.processor = CLIPProcessor.from_pretrained(config_name)
        self.text_labels = self.get_labels()
        

        self.model.to(self.device)
        # self.model.half()

        train_param = []
               
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = False

        # 解冻 CLIPVisionTransformer 后7层
        for layer in self.model.vision_model.encoder.layers[-4:]:
            for param in layer.parameters():
                train_param.append(param)

        # 解冻 CLIPVisionTransformer 的 post_layernorm 层和 proj 层
        for param in self.model.vision_model.post_layernorm.parameters():
            train_param.append(param)
        train_param.append(self.model.visual_projection)

        # 解冻 CLIPTextTransformer 的 final_layer_norm 层和 text_projection 层
        for param in self.model.text_model.final_layer_norm.parameters():
            train_param.append(param)
        train_param.append(self.model.text_projection)

        # 将所有需要训练的参数设为 requires_grad = True
        for param in train_param:
            param.requires_grad = True
        
    #训练模式
    def forward(self, data):
         # 冻结所有层
        self.model.train()
        
        images = data
        inputs = self.processor(text=self.text_labels, images=list(images), return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # inputs=inputs.half()
        # with autocast():
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        return probs
    
    #评价模式
    def get_image_features(self, data):
        self.model.eval()
            
        images = data
        
        #使用CLIP来进行图像预处理
        inputs = self.processor(text=self.text_labels, images=list(images), return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
#         with autocast():
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        return probs
    
    def get_labels(self):
        labels=[]
        file_path='./datasets/CUB_200_2011/CUB_200_2011/classes.txt'
        file_path='./datasets/Stanford_Dog/class_names.txt'
        with open(file_path, 'r') as file:
            for line in file:
                # 分割每行数据，并取第三列（索引为1）
                # parts = line.strip().split()  
                # label = parts[1][4:] if len(parts) >= 2 else None
                label=line.strip().split() [0] 
                if label:
                   
                    labels.append(label)
        return labels
    
    def save_model(self,fold_epoch=''):
        path=para_save_path+para_version+fold_epoch
        torch.save(self.model.state_dict(), path)
        print(f'VLM parameters saved to {path}')