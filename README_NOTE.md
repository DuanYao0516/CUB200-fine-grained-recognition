## Environment

1. pip uninstall tensorboard tensorboardX protobuf
2. pip install tensorboard tensorboardX

3. torch
   1. goto http://download.pytorch.org/whl/torch/
   2. find torch-1.12.0+cu102-cp37-cp37m-linux_x86_64.whl and download it
   3. upload it to server and download it
   4. pip install torch-1.12.0+cu102-cp37-cp37m-linux_x86_64.whl

4. torchvision
   1. goto http://download.pytorch.org/whl/torchvision/
   2. find torchvision-0.13.0+cu102-cp37-cp37m-linux_x86_64.whl and download it
   3. pip install torchvision-0.13.0+cu102-cp37-cp37m-linux_x86_64.whl

5. datasets
   1. goto https://data.caltech.edu/records/65de6-vp158 
   2. download the CUB_200_2011.tgz and upload it
   3. tar -xvzf CUB_200_2011.tgz 
   4. store the uncompression result into DL2024_PROJ/datasets/CUB_200_2011

6. training model's accuracy is zero
   1. in config.py, set PRE_TRAINED = True
   2. shuffle the datasets

## Various training tips and tricks   

### Momentum and Weight Decay

   in trainer.py

```python
       def trainer(self,
                train_path,
                val_path,
                label_dict,
                output_dir=None,
                log_dir=None,
                optimizer='AdamW',,
                loss_fun='Cross_Entropy',
                class_weight=None,
                lr_scheduler=None,
                cur_fold=0):
```



### Dropout

#### normal

in trainer.py 

#### attention dropout

in model/vision_transformer.py

1. **Regular Dropout**: Regular dropout is a regularization technique that reduces the complexity of a neural network and prevents overfitting by randomly setting the outputs of neurons to zero during training. By randomly dropping neuron outputs, dropout forces the model to learn more robust and generalized features.

2. **Attention Dropout**: Attention dropout is typically applied in self-attention mechanisms, such as those in Transformer models. Self-attention mechanisms are used to learn the relationships between different positions in sequence data, where attention weights determine the influence of different positions on the prediction outcome. Attention dropout introduces random dropout into the self-attention mechanism to reduce the model's dependency on specific positions, thereby increasing the model's robustness and generalization ability. In self-attention mechanisms, attention dropout is usually applied to the attention weights rather than the neuron outputs.

3. **Conclusion**: Although both involve dropping some neuron outputs, regular dropout is commonly used in traditional neural network structures such as fully connected layers or convolutional layers, whereas attention dropout is more often applied in self-attention mechanisms to enhance the performance and generalization ability of sequence models.



### BN

1. BN has already been implemented in `resnet.py`.
2. Replace it with the other three types of normalization: Layer Normalization, Instance Normalization, and Group Normalization.

```python
# Layer Normalizaiton
norm_layer = nn.LayerNorm

# instance Normalization
norm_layer = nn.InstanceNorm2d

# Group Normalization
num_groups = 4
norm_layer = nn.GroupNorm(num_groups, num_channels)
```



### K-fold Cross Validation

Adjust `config.py` to set `FOLD_NUM = 10`.

```python
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
```



### Final

in config.py

```python
FOLD_NUM = 10
batch_size = 64
'n_epoch':40,
PRE_TRAINED = True
NET_NAME = 'resnext50_32x4d'
```

in trainer.py: add AdamW

```python
       def trainer(self,
                train_path,
                val_path,
                label_dict,
                output_dir=None,
                log_dir=None,
                optimizer='AdamW',,
                loss_fun='Cross_Entropy',
                class_weight=None,
                lr_scheduler=None,
                cur_fold=0):
```

in main.py: add shuffle

```python
    if 'train' in args.mode:
        ###### modification for new data
        csv_path = './csv_file/cub_200_2011.csv_train.csv'
        label_dict = csv_reader_single(csv_path, key_col='id', value_col='label')
        path_list = list(label_dict.keys())
        np.random.seed(0)
        np.random.shuffle(path_list)
```

in main.py: change the test_csv

```python
    elif 'inf' in args.mode:

        #TODO
        test_csv_path = './csv_file/cub_200_2011.csv_test.csv'
        label_dict = csv_reader_single(test_csv_path, key_col='id', value_col='label')
        test_path = list(label_dict.keys())
        print('test len:',len(test_path))
        #########
```



### Learning Rate

在config.py中添加对应的学习率调整策略，方便进行配置

~~~python
__lr__=['ReduceLROnPlateau','MultiStepLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts','ExponentialLR']
~~~



修改train.py中的 _get_lr_scheduler：

~~~python
def _get_lr_scheduler(self, lr_scheduler, optimizer):
    if lr_scheduler == 'ReduceLROnPlateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                                  patience=5, verbose=True)
    elif lr_scheduler == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.milestones, 
                                                            gamma=self.gamma)
    elif lr_scheduler == 'CosineAnnealingLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_max)
    elif lr_scheduler == 'CosineAnnealingWarmRestarts':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 20, T_mult=2)
    elif lr_scheduler == 'ExponentialLR':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98, last_epoch=-1)

    return lr_scheduler
~~~



修改trainer中：

~~~python
for epoch in range(self.start_epoch, self.n_epoch):
    if lr_scheduler is not None:                
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            
                # 如果是 ReduceLROnPlateau 类型的学习率调度器，需要传递验证集损失作为指标参数
                    lr_scheduler.step(metrics=val_loss)
                else:
                    # 对于其他类型的学习率调度器，可以直接调用 step() 方法
                    lr_scheduler.step()
~~~



## Transfer learning  

迁移学习和学习率的修改细节。

涉及到的代码文件/文件夹：

* model/resnet.py

* config.py

* train.py

  

**以下的修改在model/resnet.py。**

迁移学习涉及到权重加载、参数冻结和参数微调。权重加载有个值得关注的点——因为分类问题的不同，全连接层的维数需要修改。以下是一些修改部分：

在文件开头添加以下内容，方便参数冻结的时候识别：

~~~python
__transfer__=['end2end','fine_tune','extract_vec']
~~~

修改resnet的统一接口，添加pretrain、transfer_type的函数参数,分别代表是否进行预训练和参数冻结的类型，进行预训练情况下和参数冻结情况下逻辑编写：

~~~python
def _resnet(arch, pretrained=False,transfer_type=None,  progress=True, **kwargs):
    model = ResNet(**kwargs)
    
    if pretrained:			#是否加载预训练的参数
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)      
        in_features = model.fc.in_features  
        out_features = model.fc.out_features 
        
        #修改全连接层的参数
        state_dict['fc.weight'] = state_dict['fc.weight'][:out_features, :in_features] 
        state_dict['fc.bias'] = state_dict['fc.bias'][:out_features] 

        # 手动设置全连接层参数为随机初始化
        for name, param in model.named_parameters():
            if 'fc' in name: 
                nn.init.normal_(param.data) 
        model.load_state_dict(state_dict)
        
        # 设置所有参数都需要梯度更新
        for param in model.parameters():
            param.requires_grad = True

        if transfer_type==__transfer__[1]
            dont_freeze_num=int(len(list(model.children()))*0.4)
            for child in list(model.children())[:dont_freeze_num]:
                for param in child.parameters():
                    param.requires_grad = False
                    
        elif transfer_type==__transfer__[2]:
             for name, param in model.named_parameters():
                if 'fc' not in name:  # 如果不是全连接层参数，则冻结
                    param.requires_grad = False

    return model
~~~

resnet的统一接口之后的各个详细接口中也要记得传递上面新增的几个参数：

~~~python
def resnet18 (pretrained=False, transfer_type=None,  #修改部分
              progress=True, **kwargs):
    return _resnet('resnet18',
                   pretrained, transfer_type,  #修改部分
                   progress,
                   block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   **kwargs)
~~~

其他同理。



**以下部分在config.py中修改。**

同样添加，方便参数配置时使用。

~~~python
__transfer__ = ['end2end','fine_tune','extract_vec']
~~~

添加两个参数

~~~python
#是否进行迁移学习
TRANSFER_LR = True
#微调类型
TRANSFER_TYPE='fine-tune'
~~~

~~~python
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
    'transfer_lr':TRANSFER_LR,		#添加
    'transfer_type':TRANSFER_TYPE,	#添加
    'vlm_trained':VLM_TRAINED,
    'vlm_ft':VLM_FT,
    'weight_path':WEIGHT_PATH,
    'weight_decay': 0.2,
    'momentum': 0.9,
    'gamma': 0.5,
    'milestones': [10,25,35,60,80],
    'T_max':10,
    'use_fp16':False,
    'dropout':0.01
 }
~~~

对应的也要在train.py增加对应的参数，在创建net的时候将参数进行传递：

~~~python
net = resnet.__dict__[net_name](
    pretrained=self.transfer_lr,  		#修改
    transfer_type=self.transfer_type,   #修改
    num_classes=self.num_classes
)
~~~



## GAN

通过在config.py设置TRANSFER_LR=True 和 TRANSFER_TYPE='fine-tune', 运行main.py可以调整是否进行迁移学习和微调。

修改SETUP_TRAINER中的'lr_scheduler'可以修改学习率调整策略。

本文档为记录开发GAN的文档。

涉及到的代码文件包括：

- model/*gan*.py
- main_GAN.py
- main_DCGAN.py
- main_ACGAN.py 


训练与生成：

运行两次

```python 
python main_DCGAN.py
```



## VIT

1. 修改config.py

   ```py
   # Arguments when trainer initial
   INIT_TRAINER = {
       'net_name':NET_NAME,
       'lr':1e-5*2, 
       'n_epoch':50,
       'num_classes':120, # CUB 200, DOG 120
       'image_size':224, # resnet 512, vit 224
       'batch_size':64,
       'train_mean':CUB_TRAIN_MEAN,
       'train_std':CUB_TRAIN_STD,
       'num_workers':2,
       'device':DEVICE,
       'pre_trained':PRE_TRAINED,
       'transfer_lr':TRANSFER_LR,
       'transfer_type':TRANSFER_TYPE,
       'weight_path':WEIGHT_PATH,
       'weight_decay': 1e-4,
       'momentum': 0.9,
       'gamma': 0.1,
       'milestones': [30,60,90],
       'T_max':5,
       'use_fp16':False,
       'dropout':0.01 
    }
   ```

2. 修改`./model/vision_transform.py`中的`VisionTransformer`类中的`init()`函数，添加微调：

   ```py
       def __init__(
           self,
           image_size: int,
           patch_size: int,
           num_layers: int,
           num_heads: int,
           hidden_dim: int,
           mlp_dim: int,
           dropout: float = 0.0,
           attention_dropout: float = 0.0,
           num_classes: int = 1000,
           representation_size: Optional[int] = None,
           norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
           conv_stem_configs: Optional[List[ConvStemConfig]] = None,
       ):
           super().__init__()
           torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
           self.image_size = image_size
           self.patch_size = patch_size
           self.hidden_dim = hidden_dim
           self.mlp_dim = mlp_dim
           self.attention_dropout = attention_dropout
           self.dropout = dropout
           self.num_classes = num_classes
           self.representation_size = representation_size
           self.norm_layer = norm_layer
   
           if conv_stem_configs is not None:
               # As per https://arxiv.org/abs/2106.14881
               seq_proj = nn.Sequential()
               prev_channels = 3
               for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                   seq_proj.add_module(
                       f"conv_bn_relu_{i}",
                       Conv2dNormActivation(
                           in_channels=prev_channels,
                           out_channels=conv_stem_layer_config.out_channels,
                           kernel_size=conv_stem_layer_config.kernel_size,
                           stride=conv_stem_layer_config.stride,
                           norm_layer=conv_stem_layer_config.norm_layer,
                           activation_layer=conv_stem_layer_config.activation_layer,
                       ),
                   )
                   prev_channels = conv_stem_layer_config.out_channels
               seq_proj.add_module(
                   "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
               )
               self.conv_proj: nn.Module = seq_proj
           else:
               self.conv_proj = nn.Conv2d(
                   in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
               )
   
           seq_length = (image_size // patch_size) ** 2
   
           # Add a class token
           self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
           seq_length += 1
   
           self.encoder = Encoder(
               seq_length,
               num_layers,
               num_heads,
               hidden_dim,
               mlp_dim,
               dropout,
               attention_dropout,
               norm_layer,
           )
           self.seq_length = seq_length
   
           heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
           if representation_size is None:
               heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
           else:
               heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
               heads_layers["act"] = nn.Tanh()
               heads_layers["head"] = nn.Linear(representation_size, num_classes)
   
           self.heads = nn.Sequential(heads_layers)
   
           if isinstance(self.conv_proj, nn.Conv2d):
               # Init the patchify stem
               fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
               nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
               if self.conv_proj.bias is not None:
                   nn.init.zeros_(self.conv_proj.bias)
           elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
               # Init the last 1x1 conv of the conv stem
               nn.init.normal_(
                   self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
               )
               if self.conv_proj.conv_last.bias is not None:
                   nn.init.zeros_(self.conv_proj.conv_last.bias)
   
           if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
               fan_in = self.heads.pre_logits.in_features
               nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
               nn.init.zeros_(self.heads.pre_logits.bias)
   
           if isinstance(self.heads.head, nn.Linear):
               nn.init.zeros_(self.heads.head.weight)
               nn.init.zeros_(self.heads.head.bias)
               
           self.train()
           for param in self.parameters():
               param.requires_grad = False
   
           # 选择性解冻后面几层
           for child in list(self.children())[-4:]:
               for param in child.parameters():
                   param.requires_grad = True
   ```

3. 修改trainer.py中`_get_net(self, net_name)`函数

   ```python
               elif net_name.startswith('vit_'):
                   import model.vision_transformer as vit
                   net = vit.__dict__[net_name](
                       #num_classes=self.num_classes,
                       image_size=self.image_size,
                       dropout=self.dropout,
                       pretrained=True
                   )
   ```

   



## VLM

关于VLM引入和参数轻量化的相关修改事宜。

涉及到的代码文件/文件夹：

* ./model
* config.py
* data_utils/data_loader.py
* train.py
* main_VLM2.py

### 环境部署

由于在学校服务器上无法直接通过url进行clip的模型的加载，于是进行了手动下载。

手动下载好的相关配置文件放在./model/b16。包括：

* config.json
* preprocessor_config.json
* pytorch_model.bin
* tokenizer_config.json
* tokenizer.json



### VLM接口的撰写

为了能够加载使用、微调VLM模型，在main_VLM2.py中实现了以下内容：

~~~python
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
~~~



### 对图像处理的修改

在图像载入对应的CLIP processor中发现了图像的处理格式不对导致模型无法正常运行的情况。解决这几点主要通过修改以下几点：

**./data_utils/data_loader.py:**将image转成RGB

~~~python
def __getitem__(self, index):
    # Get image and label
    # image: C,H,W
    # label: integer, 0,1,..
    image = cv2.imread(self.path_list[index])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  		#增加此处
~~~



**./train.py：**修改train_transformer、val_transformer、test_transformer。normalize会让数据范围不合法。

~~~python
val_transformer = transforms.Compose([
    tr.ToCVImage(),
    tr.RandomResizedCrop(size=self.image_size, scale=(1.0, 1.0)),
    tr.ToTensor(),
    # tr.Normalize(self.train_mean, self.train_std)
])
~~~



### 对config.py的修改

添加两个控制变量：vlm_trained和vlm_ft,代表是否使用vlm和是否对vlm进行微调。

对应的在train的参数列表也会增加这两项。

~~~python
#是否使用VLM
VLM_TRAINED= True

#VLM是否进行微调
VLM_FT= True
~~~

~~~python
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
~~~



### 对train.py的修改

1. VolumeClassifier中增加vlm接口的成员变量

   ~~~python
   self.vlm= VLMInterface()
   ~~~

2. 训练过程_train_on_epoch中添加使用vlm微调的过程：

   ~~~python
   if self.vlm_trained:
   	vlm_data = sample['image']
       
   	if self.vlm_ft:			#进行微调
   		output=vlm.forward(vlm_data)
           
   	else :					#不进行微调
   		output=0.4*vlm.get_image_features(vlm_data)
           
   else:
   	output = net(data)
   ~~~

   验证过程_val_on_epoch：

   ~~~python
   output = net(data)
   if self.vlm_trained:
   	vlm_data = sample['image']
       if self.vlm_ft:
           output=vlm.get_image_features(vlm_data)
       else:
           output=vlm.get_image_features(vlm_data)
   ~~~

   测试过程同理inference：

   ~~~python
   output = net(data)
   if self.vlm_trained:
       vlm_data = sample['image']
       #print(vlm_data.shape)
       output=vlm.get_image_features(vlm_data)
   ~~~

   

3. 修改def _get_optimizer，以方面对vlm中的参数进行优化：

   ~~~python
    def _get_optimizer(self, optimizer, net,vlm,lr):
           if self.vlm_trained and self.vlm_ft:
               para=vlm.model.parameters()
               optimizer = torch.optim.Adam(para, lr=lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=self.weight_decay) 
               
           else:
               para=net.parameters()
               if optimizer == 'Adam':
                   optimizer = torch.optim.Adam(para,
                                                lr=lr,
                                                weight_decay=self.weight_decay)
   
               elif optimizer == 'SGD':
                   optimizer = torch.optim.SGD(para,
                                               lr=lr,
                                               momentum=self.momentum,
                                               weight_decay=self.weight_decay)
   
               elif optimizer == 'AdamW':
                   optimizer = torch.optim.AdamW(para,
                                                lr=lr,weight_decay=self.weight_decay)
   
           return optimizer
   ~~~

4. 修改def _get_pre_trained，方便加载已经微调过的vlm参数：

   ~~~python
   def _get_pre_trained(self, weight_path):
       checkpoint = torch.load(weight_path)
       if self.vlm_trained and self.vlm_ft:
           self.vlm.model.load_state_dict(checkpoint['state_dict'])
       else:
           self.net.load_state_dict(checkpoint['state_dict'])
       self.start_epoch = checkpoint['epoch'] + 1
   ~~~



## Interpretability  

涉及到的代码文件包括：

- analisys/analysis_tools.py
- trainer.py
- main_interpretability_batch_process.py
- 本文档 interpretability.md


### 原有代码说明

原有项目代码中，已经简单实现了一个 `main_interpretability.py` 文件，首先简要介绍这个文件：

代码分析：

1. 读取特征文件、权重文件和原始图像。
2. 定义图像转换（transformation）。
3. 计算类激活图（CAMs）。
4. 保存热力图。

运行指令 

```bash
python main_interpretability.py
```

最终生成的CAM（类激活图）热力图被保存到指定的路径。具体保存路径是在 cam_path 变量中定义的。在这段代码中，cam_path 被设置为：`cam_path = './analysis/result/v1.0/'`
此外，`save_heatmap` 函数的调用：`save_heatmap(cams, img_path, class_idx, cam_path, transform=transformer)`表明生成的热力图将被保存到 `cam_path` 路径下。

从 save_heatmap 函数的参数来看：

1. cams 是计算得到的类激活图。
2. img_path 是对应的原始图像路径。
3. class_idx 是预测的类别索引。
4. cam_path 是保存热力图的路径。
5. transform是图像的变换操作。

因此，生成的CAM热力图将以文件的形式保存在路径 `./analysis/result/v1.0/` 下。具体文件名依赖于 save_heatmap 函数的实现，将会根据传入的参数进行命名和存储。


使用以下命令列出包含XXX的文件名，不包含文件夹：

```bash
ls -p | grep "Black_Footed_Albatross" | grep -v /
```

直接运行文件时出现，找不到文件：./analysis/mid_feature/v1.0/fold1/Black_Footed_Albatross_0007_796138 可能不存在
查看之后发现该文件不存在，该文件为什么不存在？因为训练集/验证集/测试集？

尝试将代码中的文件路径更改为另一张图像：Black_Footed_Albatross_0001_796111

因为原来的训练数据中，有如下文件：

```txt
root@autodl-container-43764580c0-4ffb94c8:~/autodl-tmp/jicheng1/DL2024_proj/analysis/mid_feature/v1.0/fold1# ls -p | grep "Black_Footed_Albatross" | grep -v /
Black_Footed_Albatross_0001_796111
Black_Footed_Albatross_0002_55
Black_Footed_Albatross_0003_796136
Black_Footed_Albatross_0005_796090
Black_Footed_Albatross_0006_796065
Black_Footed_Albatross_0008_796083
Black_Footed_Albatross_0016_796067
Black_Footed_Albatross_0024_796089
Black_Footed_Albatross_0025_796057
Black_Footed_Albatross_0026_796095
Black_Footed_Albatross_0033_796086
Black_Footed_Albatross_0035_796140
Black_Footed_Albatross_0037_796120
Black_Footed_Albatross_0042_796071
Black_Footed_Albatross_0045_796129
Black_Footed_Albatross_0046_18
Black_Footed_Albatross_0049_796063
Black_Footed_Albatross_0050_796125
Black_Footed_Albatross_0053_796109
Black_Footed_Albatross_0058_796074
Black_Footed_Albatross_0061_796082
Black_Footed_Albatross_0064_796101
Black_Footed_Albatross_0065_796068
Black_Footed_Albatross_0076_417
Black_Footed_Albatross_0078_796126
Black_Footed_Albatross_0082_796121
Black_Footed_Albatross_0085_92
Black_Footed_Albatross_0086_796062
Black_Footed_Albatross_0088_796133
Black_Footed_Albatross_0090_796077
```


### 新增代码说明

#### 辅助函数增加

在 `analysis.analysis_tools.py` 文件中添加函数：

修改了save_heatmap 函数，重写了一个 my_save_heatmap 

```python
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
```

#### trainer.py 的更改

我们主要的工作顺序：

1. 定义hook函数
2. 注册hook到目标层
3. 在inference中调用hook

首先，修改VolumeClassifier类，增加对其他层的hook支持：

```python
class VolumeClassifier(object):
    # ... 其他代码保持不变 ...

    def __init__(self, net_name=None, lr=1e-3, n_epoch=1, num_classes=3, image_size=None, batch_size=6, train_mean=0, train_std=0, num_workers=0, device=None, pre_trained=False, transfer_lr=False, transfer_type=None, weight_path=None, weight_decay=0., momentum=0.95, gamma=0.1, milestones=[40, 80], T_max=5, use_fp16=True, dropout=0.01):
        super(VolumeClassifier, self).__init__()

        # 前面这一大堆保持不变
        if self.pre_trained:
            self._get_pre_trained(self.weight_path)
        
        # 新增部分
        self.hooks = []

    # ... 其他代码保持不变 ...

    def hook_fn_forward(self, module, input, output):
        self.feature_in.append(input[0].cpu().numpy())
        self.feature_out.append(output.cpu().numpy())

    def add_hooks(self, layers):
        """
        Register hooks to the specified layers.
        Args:
            layers (list): List of layer names to add hooks to.
        """
        for name, layer in self.net.named_modules():
            if name in layers:
                hook = layer.register_forward_hook(self.hook_fn_forward)
                self.hooks.append(hook)

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    # 修改 inference 方法
    def inference(self, test_path, label_dict, net=None, hook_fn_forward=False, hook_layers=None): # 可解释性

        if net is None:
            net = self.net

        if hook_fn_forward and hook_layers: # 可解释性
            self.add_hooks(hook_layers)

        net = net.cuda()
        net.eval()

        test_transformer = transforms.Compose([
            tr.ToCVImage(),
            tr.RandomResizedCrop(size=self.image_size, scale=(1.0, 1.0)),
            tr.ToTensor(),
            tr.Normalize(self.train_mean, self.train_std)
        ])

        test_dataset = DataGenerator(test_path, label_dict, transform=test_transformer)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

        result = {'true': [], 'pred': [], 'prob': []}

        test_acc = AverageMeter()

        with torch.no_grad():
            for step, sample in enumerate(test_loader):
                data = sample['image']
                target = sample['label']

                data = data.cuda()
                target = target.cuda()
                
                with autocast(self.use_fp16):
                    output = net(data)
                output = F.softmax(output, dim=1)
                output = output.float()

                acc = accuracy(output.data, target)[0]
                test_acc.update(acc.item(), data.size(0))

                result['true'].extend(target.detach().tolist())
                result['pred'].extend(torch.argmax(output, 1).detach().tolist())
                result['prob'].extend(output.detach().tolist())

                print('step:{},test_acc:{:.5f}'.format(step, acc.item()))

                torch.cuda.empty_cache()

        print('average test_acc:{:.5f}'.format(test_acc.avg))

        # 清理hook 可解释性
        if hook_fn_forward:
            self.clear_hooks()

        return result, np.array(self.feature_in), np.array(self.feature_out)

    # ... 其他代码保持不变 ...

```


使用示例：（这不是我们的主要任务）

在进行测试时，可以指定要添加hook的层：

```python 
# 实例化模型
model = VolumeClassifier(net_name='resnet50', ...)
# 进行推理，并从指定层获取特征
result, feature_in, feature_out = model.inference(test_path, label_dict, hook_fn_forward=True, hook_layers=['layer3', 'layer4'])
```

在这里，hook_layers参数是一个包含层名称的列表，这些层名称需要与模型中的命名一致。例如，ResNet中的层可以是layer1，layer2，layer3，layer4等。

通过上述修改和使用方法，可以在推理过程中获取其他层的特征，并生成基于CAM的热力图。

原代码中 `main.py` 中有如下代码，尝试更改这里就可以。

```python
classifier = VolumeClassifier(**INIT_TRAINER)
# 假设要在推理时为 'layer1' 和 'layer2' 注册 hook
result, feature_in, feature_out = classifier.inference(
    test_path=test_path,
    label_dict=label_dict,
    hook_fn_forward=True,
    hook_layers=['layer1', 'layer2']
)
```

#### 新增 main_pretanility_batch_process.py

```python 
import os
import csv
import numpy as np
import pandas as pd
import data_utils.transform as tr
from config import INIT_TRAINER
from torchvision import transforms
from converter.common_utils import hdf5_reader
from analysis.analysis_tools import calculate_CAMs, my_save_heatmap
from tqdm import tqdm

            
def find_folder_with_string(root_folder, target_string):
    for root, dirs, files in os.walk(root_folder):  
        for dir in dirs:
            if target_string.lower() in dir.lower():  # 千万注意这一句，因为模型输出mid_feature时把名字改了大小写
                folder_path = os.path.join(root, dir)
                # print("Found folder with target string at: {}".format(folder_path))
                return folder_path

def load_labels_from_csv(csv_path):
    # 读取CSV文件并将其转换为字典，以图像路径为键，真实标签和预测标签为值。
    df = pd.read_csv(csv_path)
    label_dict = {}
    for _, row in df.iterrows():
        label_dict[row['path']] = {'true': row['true'], 'pred': row['pred']}
    return label_dict
            
            
def get_image_paths(features_dir):
    # 获取features_dir文件夹下所有文件的路径
    # 指定数据集路径，也就是原图路径
    img_fold_path = f'./datasets/CUB_200_2011/CUB_200_2011/images/'
    feature_files = [f for f in os.listdir(features_dir) if os.path.isfile(os.path.join(features_dir, f))]
    img_paths = []
    
    for file_name in tqdm(feature_files, total=len(feature_files), desc="Get Original Pics"):
        try:
            # 原数据集中，文件夹名字与mid_feature中的名字不一样，这个问题在 find_folder_with_string 中解决
            class_name = "_".join(file_name.split('_')[:-2])
            img_dir = find_folder_with_string(img_fold_path, class_name)
            img_path = os.path.join(img_dir, file_name + '.jpg')
            img_paths.append(img_path)
        except:
            print(f"Error processing image path for {file_name}. Skipping...")
            print(img_dir,file_name,class_name)
            # 在出现错误时，删除对应的特征文件路径
            feature_files.remove(file_name)  # 删除对应的特征文件路径
    
    return feature_files, img_paths


if __name__ == "__main__":
    # 获取中间特征、模型权重和转换器
    features_dir = './analysis/mid_feature/v1.0/fold1/'  # 需要手工指定
    weight = np.load('./analysis/result/v1.0/fold1_fc_weight.npy')  # 需要手工指定

    transformer = transforms.Compose([
        tr.ToCVImage(),
        tr.RandomResizedCrop(size=INIT_TRAINER['image_size'], scale=(1.0, 1.0)),
        tr.ToTensor(),
        tr.Normalize(INIT_TRAINER['train_mean'], INIT_TRAINER['train_std']),
        tr.ToArray(),
    ])

    # 获取所有 features 和 img 的路径,每一个feature文件对应一个路径
    feature_files, img_paths = get_image_paths(features_dir)

    csv_path = './analysis/result/v1.0/fold1.csv'  # 需要手工指定
    label_dict = load_labels_from_csv(csv_path)

    # 设置类别数与输出路径
    classes = 200
    cam_path = './analysis/result/v1.0/'  # 权重文件，需要自己指定

    for feature_file, img_path in tqdm(zip(feature_files, img_paths), total=len(feature_files), desc="Generating CAMs"):
        features = hdf5_reader(os.path.join(features_dir, feature_file), 'feature_in')
        # img_name = os.path.basename(img_path)
        if img_path in label_dict:
            true_label = label_dict[img_path]['true']
            pred_label = label_dict[img_path]['pred']
            cams = calculate_CAMs(features, weight, range(classes))
            my_save_heatmap(cams, img_path, pred_label, true_label, cam_path, transform=transformer)
        else:
            print(f"Labels not found for {img_path}. Skipping...")
```


#### 运行方法

首先无论运行 `main_pretanility.py` 还是 `main_pretanility_batch_process.py` 都需要检查代码中需要人工指定的内容，ctrl F “指定”

处理完成之后：

```bash
# 尝试运行 批处理代码
python main_pretanility_batch_process.py
```

生成的热力图存储在 代码中的 cam_path 指定的路径。

```bash
# 将生成的图像进行处理，主要是从服务器上打包下载下来，服务器上不太好处理
mkdir correct wrong
mv *Correct.jpg correct/
mv *Wrong.jpg wrong/
zip -r interpretability.zip correct wrong
mv interpretability.zip ../../../
```

从服务器上下载下来，使用python脚本中的grid（这部分代码很简单，自己随手编写一个就可以）绘制为一张图。


#### 附录

附上我自己用于处理热力图，生成grid图的脚本：

需要先阅读代码，根据代码内容，首先要创建文件夹，通过命名的方式人为手工排好图片序号。

```python
import os
from PIL import Image

def create_image_grid(rows, cols, folder_path, output_path='output.jpg'):
    # 获取文件夹中的所有图像文件
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]

    # 计算所需的图像数
    num_images = rows * cols

    # 创建空白图像网格的尺寸
    images = []
    for i in range(min(num_images, len(image_files))):
        img_path = os.path.join(folder_path, image_files[i])
        img = Image.open(img_path)
        # 调整图像大小为512x512
        img = img.resize((512, 512), Image.LANCZOS)
        images.append(img)

    if len(images) == 0:
        raise ValueError("文件夹中没有有效的图像文件。")

    # 设置单个图像的尺寸为512x512
    img_width, img_height = 512, 512

    # 创建空白大图的尺寸
    grid_width = cols * img_width
    grid_height = rows * img_height
    grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))

    # 将图像放入大图中
    for index, img in enumerate(images):
        row = index // cols
        col = index % cols
        x = col * img_width
        y = row * img_height
        grid_image.paste(img, (x, y))

    # 保存大图
    grid_image.save(output_path)
    print(f"图像网格已保存至 {output_path}")

# 示例使用
create_image_grid(4, 4, './correct/', output_path='output_correct.jpg')
# create_image_grid(3, 3, './wrong/', output_path='output_wrong.jpg')
```



## Robustness   

涉及到的代码文件包括：

- trainer.py
- main_robustness.py

### 代码说明


在 trainer.py 文件中，添加了两个对抗攻击方法（FGSM和PGD）以及一个新的推理方法 inference_with_attack。

```python
def fgsm_attack(data, epsilon, gradient):
    sign_data_grad = gradient.sign()
    perturbed_data = data + epsilon * sign_data_grad
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data


def pgd_attack(model, data, target, epsilon, alpha, num_iter):
    perturbed_data = data.clone().detach().requires_grad_(True).to(data.device)
    for _ in range(num_iter):
        output = model(perturbed_data)
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()
        gradient = perturbed_data.grad.data
        perturbed_data = perturbed_data + alpha * gradient.sign()
        perturbation = torch.clamp(perturbed_data - data, min=-epsilon, max=epsilon)
        perturbed_data = torch.clamp(data + perturbation, 0, 1).detach_()
        perturbed_data.requires_grad = True
    return perturbed_data

```



**在卷分类器类（trainer.py）中添加方法：**

```python
class VolumeClassifier:
    # ... 省略之前的代码 ...

    def inference_with_attack(self, test_path, label_dict, attack_type='FGSM', epsilon=0.3, alpha=0.01, num_iter=40, save_dir='./attack'):
        net = self.net
        net = net.cuda()
        net.eval()

        test_transformer = transforms.Compose([
            tr.ToCVImage(),
            tr.RandomResizedCrop(size=self.image_size, scale=(1.0, 1.0)),
            tr.ToTensor(),
            # tr.Normalize(self.train_mean, self.train_std)
        ])

        test_dataset = DataGenerator(test_path, label_dict, transform=test_transformer)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

        result = {'true': [], 'pred': [], 'prob': []}
        test_acc = AverageMeter()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for step, sample in enumerate(test_loader):
            data = sample['image']
            target = sample['label']
            data = data.cuda()
            target = target.cuda()

            if attack_type == 'FGSM':
                data.requires_grad = True
                output = net(data)
                loss = F.cross_entropy(output, target)
                net.zero_grad()
                loss.backward()
                data_grad = data.grad.data
                data = fgsm_attack(data, epsilon, data_grad)
            elif attack_type == 'PGD':
                data = pgd_attack(net, data, target, epsilon, alpha, num_iter)

            with autocast(self.use_fp16):
                output = net(data)

            output = F.softmax(output, dim=1)
            output = output.float()
            acc = accuracy(output.data, target)[0]
            test_acc.update(acc.item(), data.size(0))

            result['true'].extend(target.detach().tolist())
            result['pred'].extend(torch.argmax(output, 1).detach().tolist())
            result['prob'].extend(output.detach().tolist())

            # 保存攻击后的图像
            for i in range(data.size(0)):
                true_label = target[i].item()
                pred_label = torch.argmax(output[i]).item()
                img = data[i].cpu().detach()
                img_path = os.path.join(save_dir, f'{true_label}_{pred_label}_{step}_{i}.jpg')
                save_image(img, img_path)

            print('step:{}, test_acc:{:.5f}'.format(step, acc.item()))

        print('average test_acc with {} attack: {:.5f}'.format(attack_type, test_acc.avg))
        return result


```



**添加了一个入口代码，新建 main_robustness.py：**

```python
# main_robustness.py
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
    

```

### 运行方法

1. 配置环境后，先使用 `python main.py -m train/train-cross` 进行训练。
2. 填写 `main_robustness.py` 中 `weight_path` 与 `net_name`
3. 选择 `main_robustness.py` 中 `attack_type` 与 `save_dir`，请将 `save_dir` 与 `attack_type` 对应
4. 运行代码

```bash
python main_robustbness.py
```

5. 运行结果会在项目根目录/attack/<attack_type>中生成攻击生成的图片，项目根目录下生成混淆矩阵热力图“confusion_matrix”与ROC曲线图“ROC”


生成结果之后，统一把图放到 attack 文件夹中

```bash
zip -r attack.zip attack # 打包
# 下载到本地翻阅处理，建议使用FTP，比如xftp，直接 jupyter lab下载太慢了
```


### 参考文献

1. https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
2. https://blog.csdn.net/wyf2017/article/details/119676908


### 问题与解决

```bash
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /root/miniconda3/lib/python3.8/site-packages/pandas/_libs/window/aggregations.cpython-38-x86_64-linux-gnu.so)
```

参考 https://blog.csdn.net/weixin_39379635/article/details/129159713 解决



## Lightweight Model   

### 参数轻量化

确保在config.py中存在use_fp16的参数：

~~~python
'use_fp16':False,
~~~

在train.py中的_train_on_epoch：

~~~python
 with autocast(self.use_fp16):
 	if self.vlm_trained:
	vlm_data = sample['image']
    
	if self.vlm_ft:			#进行微调
		output=vlm.forward(vlm_data)
        
	else :					#不进行微调
		output=0.4*vlm.get_image_features(vlm_data)
        
    else:
        output = net(data)
~~~



~~~python
if self.use_fp16:
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    optimizer.step()
~~~



### 运行方式

通过在config.py设置vlm_train=True 和 vlm_ft=True,运行main.py就会加载微调vlm模型。最后还会保存训练过程中出现的参数。

在main_VLM.py中保存了可以通过简单的训练方式对vlm模型进行单独训练、验证和测试的方式。单独运行main_VLM.py即可。



## Empirical evaluation  

### 数据集stanford dog

1. 下载stanford dog中的lists.tar和images.tar

2. 解压到./datasets/Stanford_Dog/Images和./datasets/Stanford_Dog/lists中

3. 修改make_list

   ```python
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
   ```

   

4. 在终端运行`python make_csv.py`

5. 修改main.py

   ```python
   csv_path = './csv_file/stanford_dog.csv_train.csv'
   test_csv_path = './csv_file/stanford_dog.csv_test.csv'
   ```

6. 修改config.py

   ```python
   from utils import get_weight_path,get_weight_list
   
   __all__ = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152","resnext50_32x4d","resnext101_32x8d","resnext101_64x4d","wide_resnet50_2","wide_resnet101_2",
              "vit_b_16","vit_b_32","vit_l_16","vit_l_32","vit_h_14"]
   
   __transfer__ = ['end2end','fine_tune','extract_vec']
   
   __lr__=['ReduceLROnPlateau','MultiStepLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts','ExponentialLR']
   
   NET_NAME = 'vit_l_16'
   VERSION = 'v2.0'
   DEVICE = '0'
   # Must be True when pre-training and inference
   PRE_TRAINED = False
   
   #是否进行迁移学习
   TRANSFER_LR = True
   #微调类型
   TRANSFER_TYPE='fine-tune'
   
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
       'lr':1e-5*2, 
       'n_epoch':28,
       'num_classes':120, #CUB 200
       'image_size':224, # vit: 224, resnet: 
       'batch_size':64,
       'train_mean':CUB_TRAIN_MEAN,
       'train_std':CUB_TRAIN_STD,
       'num_workers':2,
       'device':DEVICE,
       'pre_trained':PRE_TRAINED,
       'transfer_lr':TRANSFER_LR,
       'transfer_type':TRANSFER_TYPE,
       'weight_path':WEIGHT_PATH,
       'weight_decay': 1e-4,
       'momentum': 0.9,
       'gamma': 0.1,
       'milestones': [30,60,90],
       'T_max':5,
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
   
   ```

   



+ 我们最终使用的模型架构是基于vit的VLM模型，为了测试模型的性能，我们对比了其在不同数据集上的分类性能，包括鸟类数据集CUB200 和狗类细分数据集Stanford Dog。实验结果显示，在CUB 100数据集上的准确率要显著低于Stanford Dog数据集，根据我们的分析，这是因为不同种类的狗之间有着较为显著的差异，这使得模型更容易提取出这些可以区分不同狗类的特征。而对于鸟类来说，由于其体积本来就远远小于狗，而且不同鸟类之间的差异相对于狗类来说没有那么显著，因此鸟类细分任务的难度本身是要比狗类细分任务更难的。因此，模型在CUB数据集上的表现不如Stanford Dog数据集也是易于理解的。总的来说，模型在细分类任务上表现良好。

  

### 超参数敏感性

1. 我们还进行了超参数敏感性实验，我们调整的参数包括学习率、fold num等一系列参数，下面主要列出对程序性能有显著影响的超参数：第一个是学习率，尽管我们使用了动态学习率调整策略，但初始学习率的设置一样非常重要，我们尝试了许多学习率学习率，当我们设置的学习率较大（数量级在e-3）时，不仅最后的准确率不高，而且收敛速度也比较慢，训练过程中也会出现不稳定的情况。而学习率设置的较小（e-6）时，收敛速度会非常慢，而且容易陷入局部最优，且泛化效果不佳，导致最终准确率不高。最后选择了lr = 2*e-5，此时收敛速度快，训练过程稳定，且结果准确率高，模型泛化效果好。
2. 第二个是Fold num，我们分别尝试了FOLD NUM= 5和FOLD NUM = 10，发现FOLD NUM 为 10时的准确率明显要高于为5时，这是因为FOLD NUM = 10时，每次训练的数据量更大，由于我们的数据量较小，因此这一点尤为关键。此外，更多的fold num意味着更多次的训练和验证循环。这样可以更细致地评估模型的性能，减少由于数据分割方式引起的随机误差。通过多次验证，可以更全面地评估模型的稳定性和泛化能力，从而获得更准确的性能评估结果。此外，较小的数据集可能导致模型过拟合，如果fold数量少，每次训练的数据量较少，模型可能会过拟合到训练数据上的噪音。增加fold数量意味着每次训练时使用更多的数据，从而减少过拟合的风险，提高模型的泛化能力。虽然为5时计算速度更快，开销更小。但由于我们的任务中数据集较小，且细分类任务较为细致和复杂，因此FOld NUM=10和我们的任务更为契合，最后我们选择了Fold num = 10。
3. 迁移学习时lr = 5 * e-6时效果最好，lr = 2 * e-3时训练效果不佳，因为步幅太大，导致参数变化过快，实际上进行迁移学习后的参数都是比较好的了，这时不宜进行大幅度的改动，应该让学习率尽量小，但学习率太小也会导致收敛速度过慢，训练时间太长
4. Fold_num = 10时效果最好，让训练集尽可能大，但也应该保留一定数量的数据作为验证集，使得验证集结果有一定的代表性
5. 'dropout' = 0.01时效果最好，相当于不适用dropout时效果最好，此时可能没有过拟合问题



## Ensemble Learning

1. 在主文件夹下添加文件`ensemble_learning.py`:

   ```py
   import os
   import pandas as pd
   from collections import defaultdict
   from data_utils.csv_reader import csv_reader_single
   
   # Example usage
   weights = {
       "resnet18": 0.8,
       "resnet34": 0.85,
       "resnet50": 0.9,
       "resnet101": 0.92,
       "resnet152": 0.95,
       "resnext50_32x4d": 0.87,
       "resnext101_32x8d": 0.89,
       "resnext101_64x4d": 0.91,
       "wide_resnet50_2": 0.86,
       "wide_resnet101_2": 0.88,
       "vit_b_16": 0.84,
       "vit_b_32": 0.82,
       "vit_l_16": 0.739,
       "vit_l_32": 0.81,
       "vit_h_14": 0.79
   }
   
   def ensemble_predictions(result_dir):
       # Initialize vote dictionary
       vote = defaultdict(lambda: defaultdict(float))
   
       true_labels_dict = {}
       first_file = True
       
       # Read all files from result_dir
       for file in os.listdir(result_dir):
           if file.endswith("fold1.csv"):
               # Get model name from file name
               NET_NAME = file.split('.')[0]
               
               # Get the weight for the current model
               weight = weights.get(NET_NAME, 0)
               
               # Read the CSV file predictions
               file_path = os.path.join(result_dir, file)
               file_csv = pd.read_csv(file_path)
               
               if first_file:
                   # Read TRUE labels only once
                   true_labels_dict = csv_reader_single(file_path, key_col="path", value_col="TRUE")
                   first_file = False
               
               predictions = csv_reader_single(file_path, key_col="path", value_col="pred")
               
               # Update the vote dictionary
               for path, pred in predictions.items():
                   vote[path][pred] += weight
       
       # Get final prediction based on votes
       final_predictions = []
       true_labels = []
       for path in sorted(vote.keys()):  # Ensure consistent order
           pred_dict = vote[path]
           final_predictions.append(max(pred_dict, key=pred_dict.get))
           true_labels.append(true_labels_dict[path])
       
       result = {
           'true': true_labels,
           'pred': final_predictions
       }
       
       return result
   
   
   
   result_dir = './analysis/result/v2.0'
   final_predictions = ensemble_predictions(result_dir)
   
   # Example of printing the final predictions
   for path, pred in final_predictions.items():
       print(f"{path}: {pred}")
   ```

   

2. 在main中添加`from ensemble_learning import ensemble_predictions`

3. 修改main中测试时的保存路径`save_path = os.path.join(save_dir,f'{str(INIT_TRAINER['net_name'])}.fold{str(CURRENT_FOLD)}.csv')`

4. 修改main中测试时（即`inf`分支处）, 在`result['path'] = test_path`语句前更新result = ensemble_predictions('./analysis/result/v2.0')

