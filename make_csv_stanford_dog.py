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
import glob
import pandas as pd

import os
import glob
import pandas as pd
from scipy.io import loadmat

def make_csv_cub(input_path, csv_path, index_path=None):
    '''
    Make CUB 200 2011 csv file.
    '''
    
    info = []
    for subdir in os.scandir(input_path):
        if subdir.is_dir():
            path_list = glob.glob(os.path.join(subdir.path, "*.jpg"))
            sub_info = [[item] for item in path_list]
            info.extend(sub_info)

    col = ['id']
    info_data = pd.DataFrame(columns=col, data=info)
    
    if index_path is not None:
        # Load the .mat file
        mat_data = loadmat(index_path)
        
        # Check if 'labels' is present in the .mat file
        if 'labels' not in mat_data:
            raise KeyError('The provided .mat file does not contain the variable "labels".')
        
        # Extract and format labels
        labels = mat_data['labels'].squeeze()  # Convert to 1D array if necessary
        
        # Ensure length of labels matches info_data
        if len(labels) != len(info_data):
            raise ValueError('Length of labels in .mat file does not match number of images.')
        
        # Assign labels to info_data
        info_data['label'] = labels - 1
    
    info_data.to_csv(csv_path, index=False)

    
def split_csv_cub(input_path, split_train_path, split_test_path, csv_path):
    '''
    Split CUB 200 2011 csv file using .mat files for train and test splits.
    '''
    
    # Load the .mat files
    train_data = loadmat(split_train_path)
    test_data = loadmat(split_test_path)
    
    # Extract the file lists (indices) and labels from the .mat files
    if 'file_list' not in train_data or 'file_list' not in test_data:
        raise KeyError('The provided .mat files do not contain the variable "file_list".')
    
    train_files = train_data['file_list'].squeeze()  # Convert to 1D array if necessary
    train_files = [os.path.join(input_path, i[0]) for i in train_files]  
    train_labels = train_data['labels'].squeeze()
    
    test_files = test_data['file_list'].squeeze()  # Convert to 1D array if necessary
    test_files = [os.path.join(input_path, i[0]) for i in test_files]
    test_labels = test_data['labels'].squeeze()
    
    # Load the CSV file
    info_data = pd.read_csv(csv_path)
    
    # Assign labels to the CSV data
    info_data['label'] = -1  # Initialize labels column
    
    # Assign labels to training data
    for file, label in zip(train_files, train_labels):
        info_data.loc[info_data['id'] == file, 'label'] = label - 1
    train_data = info_data[info_data['label'] != -1]  # Corrected condition
    
    info_data['label'] = -1
    # Assign labels to test data
    for file, label in zip(test_files, test_labels):
        info_data.loc[info_data['id'] == file, 'label'] = label - 1
    
    # Split the data based on the file lists
    
    test_data = info_data[info_data['label'] != -1]   # Corrected condition
    
    # Save the split data to new CSV files
    train_data.to_csv(csv_path + '_train.csv', index=False)
    test_data.to_csv(csv_path + '_test.csv', index=False)



if __name__ == "__main__":
    
    # make csv file
    if not os.path.exists('./csv_file'):
        os.makedirs('./csv_file')

    input_path = './datasets/Stanford_Dog/Images/'
    csv_path = './csv_file/stanford_dog.csv'
    index_path = './datasets/Stanford_Dog/lists/file_list.mat'
    make_csv_cub(input_path, csv_path, index_path)
    
    # split csv file
    split_train_path = './datasets/Stanford_Dog/lists/train_list.mat'
    split_test_path = './datasets/Stanford_Dog/lists/test_list.mat'
    split_csv_cub(input_path, split_train_path, split_test_path, csv_path)
    

    
