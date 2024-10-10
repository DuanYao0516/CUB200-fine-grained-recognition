from utils import get_weight_path,get_weight_list

__all__ = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152","resnext50_32x4d","resnext101_32x8d","resnext101_64x4d","wide_resnet50_2","wide_resnet101_2",
           "vit_b_16","vit_b_32","vit_l_16","vit_l_32","vit_h_14"]

__transfer__ = ['end2end','fine_tune','extract_vec']

__lr__=['ReduceLROnPlateau','MultiStepLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts','ExponentialLR']

NET_NAME = 'vit_b_32'
VERSION = 'v2.0'
DEVICE = '0'
# Must be True when pre-training and inference
PRE_TRAINED = False

#是否进行迁移学习
TRANSFER_LR = True
#微调类型
TRANSFER_TYPE='fine-tune'
#是否使用VLM
VLM_TRAINED= True
#VLM是否进行微调
VLM_FT= True

# 1,2,3,4,5
CURRENT_FOLD = 1
GPU_NUM = len(DEVICE.split(','))
FOLD_NUM = 10

CUB_TRAIN_MEAN = [0.48560741861744905, 0.49941626449353244, 0.43237713785804116]
CUB_TRAIN_STD = [0.2321024260764962, 0.22770540015765814, 0.2665100547329813]

CKPT_PATH = './ckpt/{}/fold{}'.format(VERSION,CURRENT_FOLD)
WEIGHT_PATH = get_weight_path(CKPT_PATH)
# print(WEIGHT_PATH)

if PRE_TRAINED:
    WEIGHT_PATH_LIST = get_weight_list('./ckpt/{}/'.format(VERSION))
else:
    WEIGHT_PATH_LIST = None

# Arguments when trainer initial
INIT_TRAINER = {
    'net_name':NET_NAME,
    'lr':5e-6, 
    'n_epoch':100,
    'num_classes':120,
    'image_size':224,
    'batch_size':64,
    'train_mean':CUB_TRAIN_MEAN,
    'train_std':CUB_TRAIN_STD,
    'num_workers':2,
    'device':DEVICE,
    'pre_trained':PRE_TRAINED,
    'transfer_lr':TRANSFER_LR,
    'transfer_type':TRANSFER_TYPE,
    'vlm_trained':VLM_TRAINED,		#增加
    'vlm_ft':VLM_FT,				#增加
    'weight_path':WEIGHT_PATH,
    'weight_decay': 0.2,
    'momentum': 0.9,
    'gamma': 0.5,
    'milestones': [10,25,35,60,80],
    'T_max':10,
    'use_fp16':False,
    'dropout':0.01
 }

# Arguments when perform the trainer 
SETUP_TRAINER = {
    'output_dir':'./ckpt/{}'.format(VERSION),
    'log_dir':'./log/{}'.format(VERSION),
    'optimizer':'AdamW',
    'loss_fun':'Cross_Entropy',
    'class_weight':None,
    'lr_scheduler':'MultiStepLR' 
}

