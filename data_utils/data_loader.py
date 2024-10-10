import sys
import cv2
from torch.utils.data import Dataset
import torch


class DataGenerator(Dataset):
    '''
    Custom Dataset class for data loader.
    Args：
    - path_list: list of file path
    - label_dict: dict, file path as key, label as value
    - transform: the data augmentation methods
    '''
    def __init__(self, path_list, label_dict, transform=None):

        self.path_list = path_list
        self.label_dict = label_dict
        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        # Get image and label
        # image: C,H,W
        # label: integer, 0,1,..
        image = cv2.imread(self.path_list[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        
        # Transform
        if self.transform is not None:
            image = self.transform(image)
        
        label = self.label_dict[self.path_list[index]]
        sample = {'image': image, 'label': int(label)}


        return sample

