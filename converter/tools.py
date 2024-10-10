import os
import glob
import pandas as pd
import random
from tqdm import tqdm

from common_utils import hdf5_reader

RULE = {"non-r0": 0, "r0": 1}


def make_label_csv(input_path, csv_path):
    '''
    Make label csv file.
    label rule: non-r0->0, r0->1
    '''
    info = []
    for subdir in os.scandir(input_path):
        index = RULE[subdir.name]
        path_list = glob.glob(os.path.join(subdir.path, "*.hdf5"))
        sub_info = [[item, index] for item in path_list]
        info.extend(sub_info)

    col = ['id', 'label']
    random.shuffle(info)
    info_data = pd.DataFrame(columns=col, data=info)
    info_data.to_csv(csv_path, index=False)


def statistic_slice_num(input_path, csv_path):
    '''
    Count the slice number for per sample.
    '''
    info = []
    for subdir in os.scandir(input_path):
        path_list = glob.glob(os.path.join(subdir.path, "*.hdf5"))
        sub_info = [[item, hdf5_reader(item, 'image').shape[0]] for item in path_list]
        info.extend(sub_info)

    col = ['id', 'slice_num']
    info_data = pd.DataFrame(columns=col, data=info)
    info_data.to_csv(csv_path, index=False)




if __name__ == "__main__":
    
    # Part-1: make label csv file
    #os.makedirs('./csv_file')

    input_path = os.path.abspath('../dataset/npy_data/full_data')
    csv_path = './csv_file/full_index.csv'
    make_label_csv(input_path,csv_path)
    
    
    # Part-2: Count the slice number
    # input_path = os.path.abspath('../dataset/npy_data/')
    # csv_path = './csv_file/slice_number.csv'
    # statistic_slice_num(input_path,csv_path)
    
