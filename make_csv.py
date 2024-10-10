import os
import glob
import pandas as pd

def make_csv_cub(input_path, csv_path, index_path = None):
    '''
    Make CUB 200 2011 csv file.
    '''
    
    info = []
    for subdir in os.scandir(input_path):
        label = int(subdir.name.split('.')[0])
        path_list = glob.glob(os.path.join(subdir.path, "*.jpg"))
        sub_info = [[item, label] for item in path_list]
        info.extend(sub_info)

    col = ['id', 'label']
    info_data = pd.DataFrame(columns=col, data=info)
    info_data['label']= info_data['label'] -1
    
    if index_path is not None:
        index = pd.read_csv(index_path, header=None, sep=' ').loc[:,1]
        index = input_path + index
        info_data.index = info_data['id']
        info_data = info_data.loc[index]
        
    info_data.to_csv(csv_path, index=False)
    
def split_csv_cub(split_path, csv_path):
    '''
    Split CUB 200 2011 csv file.
    '''
    
    split = pd.read_csv(split_path, header=None, sep=' ').loc[:,1]
    info_data = pd.read_csv(csv_path)
 
    train_data = info_data.loc[split == 1]
    test_data = info_data.loc[split == 0]
    
    train_data.to_csv(csv_path + '_train.csv', index=False)
    test_data.to_csv(csv_path + '_test.csv', index=False)

if __name__ == "__main__":
    
    # make csv file
    if not os.path.exists('./csv_file'):
        os.makedirs('./csv_file')

    input_path = './datasets/CUB_200_2011/CUB_200_2011/images/'
    csv_path = './csv_file/cub_200_2011.csv'
    index_path = './datasets/CUB_200_2011/CUB_200_2011/images.txt'
    make_csv_cub(input_path, csv_path, index_path)
    
    # split csv file
    split_path = './datasets/CUB_200_2011/CUB_200_2011/train_test_split.txt'
    split_csv_cub(split_path, csv_path)
    

    
