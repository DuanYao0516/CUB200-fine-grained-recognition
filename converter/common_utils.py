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
import h5py
import numpy as np
import glob

def save_as_hdf5(data, save_path, key=None):
    '''
    Numpy array save as hdf5.

    Args:
    - data: numpy array
    - save_path: string, destination path
    - key: string, key value for reading
    '''
    hdf5_file = h5py.File(save_path, 'a')
    hdf5_file.create_dataset(key, data=data)
    hdf5_file.close()

def hdf5_reader(data_path, key=None):
    '''
    Hdf5 file reader, return numpy array.
    '''
    hdf5_file = h5py.File(data_path,'r')
    image = np.asarray(hdf5_file[key],dtype=np.float32)
    hdf5_file.close()

    return image


