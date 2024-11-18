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
import argparse
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from data_utils.csv_reader import csv_reader_single
from ensemble_learning import ensemble_predictions
import time
import shutil

from trainer import VolumeClassifier
from data_utils.csv_reader import csv_reader_single
from config import INIT_TRAINER, SETUP_TRAINER, VERSION, CURRENT_FOLD, WEIGHT_PATH_LIST,FOLD_NUM
from converter.common_utils import save_as_hdf5


def get_cross_validation(path_list, fold_num, current_fold):

    _len_ = len(path_list) // fold_num
    train_id = []
    validation_id = []
    end_index = current_fold * _len_
    start_index = end_index - _len_
    if current_fold == fold_num:
        validation_id.extend(path_list[start_index:])
        train_id.extend(path_list[:start_index])
    else:
        validation_id.extend(path_list[start_index:end_index])
        train_id.extend(path_list[:start_index])
        train_id.extend(path_list[end_index:])

    print(f'train sample number:{len(train_id)}, val sample number:{len(validation_id)}')
    return train_id, validation_id


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--mode',
                        default='train',
                        choices=["train-cross","train", "inf","inf-cross"],
                        help='choose the mode',
                        type=str)
    parser.add_argument('-s',
                        '--save',
                        default='yes',
                        choices=['no', 'n', 'yes', 'y'],
                        help='save the forward middle features or not',
                        type=str)
    parser.add_argument('-n',
                        '--net_name',
                        #default=None,
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152","resnext50_32x4d","resnext101_32x8d","resnext101_64x4d","wide_resnet50_2","wide_resnet101_2",
           "vit_b_16","vit_b_32","vit_l_16","vit_l_32","vit_h_14"],
                        help='override the INIT_TRAINER[\'net_name\'] of config.py',
                        type=str)
    parser.add_argument('-l',
                        '--lr',
                        #default=None,
                        help='override the INIT_TRAINER[\'lr\'] of config.py',
                        type=float)
    parser.add_argument('-e',
                        '--n_epoch',
                        #default=None,
                        help='override the INIT_TRAINER[\'n_epoch\'] of config.py',
                        type=int)
    parser.add_argument('-c',
                        '--num_classes',
                        #default=None,
                        help='override the INIT_TRAINER[\'num_classes\'] of config.py',
                        type=int)
    parser.add_argument('-is',
                        '--image_size',
                        #default=None,
                        help='override the INIT_TRAINER[\'image_size\'] of config.py',
                        type=int)
    parser.add_argument('-bs',
                        '--batch_size',
                        #default=None,
                        help='override the INIT_TRAINER[\'batch_size\'] of config.py',
                        type=int)
    args = parser.parse_args()
    
    if args.net_name is not None:
        INIT_TRAINER['net_name'] = args.net_name
    if args.lr is not None:
        INIT_TRAINER['lr'] = args.lr
    if args.n_epoch is not None:
        INIT_TRAINER['n_epoch'] = args.n_epoch
    if args.num_classes is not None:
        INIT_TRAINER['num_classes'] = args.num_classes
    if args.image_size is not None:
        INIT_TRAINER['image_size'] = args.image_size
    if args.batch_size is not None:
        INIT_TRAINER['batch_size'] = args.batch_size

    # Set data path & classifier
    
    if args.mode != 'train-cross' and args.mode != 'inf-cross':
        classifier = VolumeClassifier(**INIT_TRAINER)
        print(get_parameter_number(classifier.net))

    # Training
    ###############################################
    if 'train' in args.mode:
        ###### modification for new data
        # csv_path = './csv_file/cub_200_2011.csv_train.csv'
        csv_path = './csv_file/stanford_dog.csv_train.csv'
        label_dict = csv_reader_single(csv_path, key_col='id', value_col='label')
        path_list = list(label_dict.keys())
        np.random.seed(0)
        np.random.shuffle(path_list)

        if args.mode == 'train-cross':
            for fold in range(1,FOLD_NUM+1):
                print('===================fold %d==================='%(fold))
                if INIT_TRAINER['pre_trained']:
                    INIT_TRAINER['weight_path'] = WEIGHT_PATH_LIST[fold-1]
                classifier = VolumeClassifier(**INIT_TRAINER)
                train_path, val_path = get_cross_validation(path_list, FOLD_NUM, fold)
                SETUP_TRAINER['train_path'] = train_path
                SETUP_TRAINER['val_path'] = val_path
                SETUP_TRAINER['label_dict'] = label_dict
                SETUP_TRAINER['cur_fold'] = fold

                start_time = time.time()
                classifier.trainer(**SETUP_TRAINER)

                print('run time:%.4f' % (time.time() - start_time))
        
        elif args.mode == 'train':
            train_path, val_path = get_cross_validation(path_list, FOLD_NUM, CURRENT_FOLD)
            SETUP_TRAINER['train_path'] = train_path
            SETUP_TRAINER['val_path'] = val_path
            SETUP_TRAINER['label_dict'] = label_dict
            SETUP_TRAINER['cur_fold'] = CURRENT_FOLD

            start_time = time.time()
            classifier.trainer(**SETUP_TRAINER)

            print('run time:%.4f' % (time.time() - start_time))
    ###############################################

    # Inference
    ###############################################
    elif 'inf' in args.mode:

        #TODO
        # test_csv_path = './csv_file/cub_200_2011.csv_test.csv'
        test_csv_path = './csv_file/stanford_dog.csv_test.csv'
        label_dict = csv_reader_single(test_csv_path, key_col='id', value_col='label')
        test_path = list(label_dict.keys())
        print('test len:',len(test_path))
        #########

        save_dir = './analysis/result/{}'.format(VERSION)
        feature_dir = './analysis/mid_feature/{}'.format(VERSION)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        if args.mode == 'inf':
            save_path = os.path.join(save_dir,f'fold{str(CURRENT_FOLD)}.csv')
            start_time = time.time()
            if args.save == 'no' or args.save == 'n':
                result, _, _ = classifier.inference(test_path, label_dict)
                print('run time:%.4f' % (time.time() - start_time))
            else:
                result, feature_in, feature_out = classifier.inference(
                    test_path, label_dict, hook_fn_forward=True)
                print('run time:%.4f' % (time.time() - start_time))
                # save the avgpool output
                print(feature_in.shape, feature_out.shape)
                feature_save_path = os.path.join(feature_dir,f'fold{str(CURRENT_FOLD)}')
                if not os.path.exists(feature_save_path):
                    os.makedirs(feature_save_path)
                else:
                    shutil.rmtree(feature_save_path)
                    os.makedirs(feature_save_path)
                for i in range(len(test_path)):
                    name = os.path.basename(test_path[i])
                    feature_path = os.path.join(feature_save_path, name.split(".")[0])
                    save_as_hdf5(feature_in[i], feature_path, 'feature_in')
                    save_as_hdf5(feature_out[i], feature_path, 'feature_out')
            result['path'] = test_path
            csv_file = pd.DataFrame(result)
            csv_file.to_csv(save_path, index=False)
            #report
            cls_report = classification_report(
                result['true'],
                result['pred'],
                output_dict=True)
            #fc weight
            if INIT_TRAINER['net_name'].startswith('res') or INIT_TRAINER['net_name'].startswith('wide_res'):

                fc_weight_save_path = os.path.join(save_dir,f'fold{str(CURRENT_FOLD)}_fc_weight.npy')
                np.save(fc_weight_save_path, classifier.net.state_dict()['fc.weight'].cpu().numpy())
            
            #save as csv
            report_save_path = os.path.join(save_dir,f'fold{str(CURRENT_FOLD)}_report.csv')
            report_csv_file = pd.DataFrame(cls_report)
            report_csv_file.to_csv(report_save_path)
        

        elif args.mode == 'inf-cross':

            for fold in range(1,FOLD_NUM+1):
                print('===================fold %d==================='%(fold))
                print('weight path %s'%WEIGHT_PATH_LIST[fold-1])
                INIT_TRAINER['weight_path'] = WEIGHT_PATH_LIST[fold-1]
                classifier = VolumeClassifier(**INIT_TRAINER)
                
                save_path = os.path.join(save_dir,f'fold{str(fold)}.csv')
                start_time = time.time()
                if args.save == 'no' or args.save == 'n':
                    result, _, _ = classifier.inference(test_path, label_dict)
                    print('run time:%.4f' % (time.time() - start_time))
                else:
                    result, feature_in, feature_out = classifier.inference(
                        test_path, label_dict, hook_fn_forward=True)
                    print('run time:%.4f' % (time.time() - start_time))
                    # save the avgpool output
                    print(feature_in.shape, feature_out.shape)
                    feature_save_path = os.path.join(feature_dir,f'fold{str(fold)}')
                    if not os.path.exists(feature_save_path):
                        os.makedirs(feature_save_path)
                    else:
                        shutil.rmtree(feature_save_path)
                        os.makedirs(feature_save_path)
                    for i in range(len(test_path)):
                        name = os.path.basename(test_path[i])
                        feature_path = os.path.join(feature_save_path, name.split(".")[0])
                        save_as_hdf5(feature_in[i], feature_path, 'feature_in')
                        save_as_hdf5(feature_out[i], feature_path, 'feature_out')
                result['path'] = test_path
                csv_file = pd.DataFrame(result)
                csv_file.to_csv(save_path, index=False)
                #report
                cls_report = classification_report(
                    result['true'],
                    result['pred'],
                    output_dict=True)
                
                #fc weight
                if INIT_TRAINER['net_name'].startswith('res') or INIT_TRAINER['net_name'].startswith('wide_res'):
                    fc_weight_save_path = os.path.join(save_dir,f'fold{str(fold)}_fc_weight.npy')
                    np.save(fc_weight_save_path, classifier.net.state_dict()['fc.weight'].cpu().numpy())
 
                
                #save as csv
                report_save_path = os.path.join(save_dir,f'fold{str(fold)}_report.csv')
                report_csv_file = pd.DataFrame(cls_report)
                report_csv_file.to_csv(report_save_path)
    ###############################################
