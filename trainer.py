import os
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms
import numpy as np
import math
import shutil
from torchvision import transforms
from torch.nn import functional as F

import data_utils.transform as tr
from data_utils.data_loader import DataGenerator

from utils import dfs_remove_weight
from torch.cuda.amp import autocast, GradScaler
# GPU version.
from main_VLM2 import VLMInterface

class VolumeClassifier(object):
    '''
    Control the training, evaluation, and inference process.
    Args:
    - net_name: string, __all__ = ["resnet18", "resnet34", "resnet50",...].
    - lr: float, learning rate.
    - n_epoch: integer, the epoch number
    - num_classes: integer, the number of class
    - image_size: integer, input size
    - batch_size: integer
    - num_workers: integer, how many subprocesses to use for data loading.
    - device: string, use the specified device
    - pre_trained: True or False, default False
    - weight_path: weight path of pre-trained model
    '''
    def __init__(self,
                 net_name=None,
                 lr=1e-3,
                 n_epoch=1,
                 num_classes=3,
                 image_size=None,
                 batch_size=6,
                 train_mean=0,
                 train_std=0,
                 num_workers=0,
                 device=None,
                 pre_trained=False,
                 transfer_lr=False,
                 transfer_type=None,
                 vlm_trained=False,
                 vlm_ft=False,
                 weight_path=None,
                 weight_decay=0.,
                 momentum=0.95,
                 gamma=0.1,
                 milestones=[40, 80],
                 T_max=5,
                 use_fp16=True,
                 dropout=0.01):
        super(VolumeClassifier, self).__init__()

        self.net_name = net_name
        self.lr = lr
        self.n_epoch = n_epoch
        self.num_classes = num_classes
        self.image_size = image_size
        self.batch_size = batch_size
        self.train_mean = train_mean
        self.train_std = train_std
        
        self.num_workers = num_workers
        self.device = device

        self.pre_trained = pre_trained
        self.transfer_lr = transfer_lr
        self.transfer_type = transfer_type
        self.vlm_trained=vlm_trained
        self.vlm_ft=vlm_ft
        self.weight_path = weight_path
        self.start_epoch = 0
        self.global_step = 0
        self.loss_threshold = 1.0
        self.metric_threshold = 0.7
        # save the middle output
        self.feature_in = []
        self.feature_out = []

        self.weight_decay = weight_decay
        self.momentum = momentum
        self.gamma = gamma
        self.milestones = milestones
        self.T_max = T_max
        self.use_fp16 = use_fp16
        self.dropout = dropout

        os.environ['CUDA_VISIBLE_DEVICES'] = self.device

        self.net = self._get_net(self.net_name)
        self.vlm= VLMInterface()
        if self.pre_trained:
            self._get_pre_trained(self.weight_path)

    def trainer(self,
                train_path,
                val_path,
                label_dict,
                output_dir=None,
                log_dir=None,
                optimizer='AdamW',
                loss_fun='Cross_Entropy',
                class_weight=None,
                lr_scheduler=None,
                cur_fold=0):

        torch.manual_seed(0)
        np.random.seed(0)
        torch.cuda.manual_seed_all(0)
        print('Device:{}'.format(self.device))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        log_dir = os.path.join(log_dir, f'fold{str(cur_fold)}')
        output_dir = os.path.join(output_dir, f'fold{str(cur_fold)}')

        if os.path.exists(log_dir):
            if not self.pre_trained:
                shutil.rmtree(log_dir)
                os.makedirs(log_dir)
        else:
            os.makedirs(log_dir)

        if os.path.exists(output_dir):
            if not self.pre_trained:
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)
        else:
            os.makedirs(output_dir)

        self.writer = SummaryWriter(log_dir)
        self.global_step = self.start_epoch * math.ceil(
            len(train_path) / self.batch_size)

        net = self.net
        lr = self.lr
        loss = self._get_loss(loss_fun, class_weight)

        if len(self.device.split(',')) > 1:
            net = DataParallel(net)

        # dataloader setting
        train_transformer = transforms.Compose([
            tr.ToCVImage(),
            tr.RandomResizedCrop(size=self.image_size, scale=(1.0, 1.0)),
            tr.ToTensor(),
#             tr.Normalize(self.train_mean, self.train_std)
        ])

        train_dataset = DataGenerator(train_path,
                                      label_dict,
                                      transform=train_transformer)

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  pin_memory=True)

        # copy to gpu
        net = net.cuda()
        loss = loss.cuda()
        
    #是否使用到vlm
        # if self.vlm_trained:
        #     if self.vlm_ft:
        #         self.vlm = VLMInterface()#创建vlm接口
        #     else:
        #         self.vlm = VLMInterface(load_pretrained=True)#创建vlm接口

            
        # optimizer setting
        optimizer = self._get_optimizer(optimizer, net,self.vlm, lr)
        scaler = GradScaler()

        if self.pre_trained:
            checkpoint = torch.load(self.weight_path)
            optimizer.load_state_dict(checkpoint['optimizer'])

        if lr_scheduler is not None:
            lr_scheduler = self._get_lr_scheduler(lr_scheduler, optimizer)

        early_stopping = EarlyStopping(patience=50,
                                       verbose=True,
                                       monitor='val_acc',
                                       best_score=self.metric_threshold,
                                       op_type='max')

        for epoch in range(self.start_epoch, self.n_epoch):

            train_loss, train_acc = self._train_on_epoch(epoch, net,self.vlm, loss, optimizer, train_loader, scaler)

            torch.cuda.empty_cache()

            val_loss, val_acc = self._val_on_epoch(epoch, net, self.vlm, loss, val_path, label_dict)

            if lr_scheduler is not None:
                
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # 如果是 ReduceLROnPlateau 类型的学习率调度器，需要传递验证集损失作为指标参数
                    lr_scheduler.step(metrics=val_loss)
                else:
                    # 对于其他类型的学习率调度器，可以直接调用 step() 方法
                    lr_scheduler.step()

            print('epoch:{},train_loss:{:.5f},val_loss:{:.5f}'.format(epoch, train_loss, val_loss))

            print('epoch:{},train_acc:{:.5f},val_acc:{:.5f}'.format(epoch, train_acc, val_acc))

            self.writer.add_scalars('data/loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            self.writer.add_scalars('data/acc', {
                'train': train_acc,
                'val': val_acc
            }, epoch)
            self.writer.add_scalar('data/lr', optimizer.param_groups[0]['lr'],epoch)

            early_stopping(val_acc)

            if val_acc > self.metric_threshold:
                self.metric_threshold = val_acc
                if self.vlm_trained and self.vlm_ft:
                    if len(self.device.split(',')) > 1:
                        state_dict = self.vlm.model.state_dict()
                    else:
                        state_dict = self.vlm.state_dict()
                #     # pass
                #     self.vlm.save_model('/epoch={}-train_loss={:.5f}-val_loss={:.5f}-train_acc={:.5f}-val_acc={:.5f}.pth'.format(epoch, train_loss, val_loss, train_acc, val_acc))
                else:
                    if len(self.device.split(',')) > 1:
                        state_dict = net.module.state_dict()
                    else:
                        state_dict = net.state_dict()

                saver = {
                    'epoch': epoch,
                    'save_dir': output_dir,
                    'state_dict': state_dict,
                    'optimizer': optimizer.state_dict()
                }

                file_name = 'epoch={}-train_loss={:.5f}-val_loss={:.5f}-train_acc={:.5f}-val_acc={:.5f}.pth'.format(
                    epoch, train_loss, val_loss, train_acc, val_acc)
                print('Save as :', file_name)
                save_path = os.path.join(output_dir, file_name)

                torch.save(saver, save_path)

            #early stopping
            if early_stopping.early_stop:
                print('Early Stopping!')
                break
            

        self.writer.close()
        dfs_remove_weight(output_dir, 5)

    def _train_on_epoch(self, epoch, net,vlm ,criterion, optimizer, train_loader,scaler):
        
        net.train()
        
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        
        for step, sample in enumerate(train_loader):

            data = sample['image']
            target = sample['label']

            data = data.cuda()
            target = target.cuda()
            
            with autocast(self.use_fp16):
                
                if self.vlm_trained:
                    vlm_data = sample['image']
                    if self.vlm_ft:
                        output=vlm.forward(vlm_data)
                        
                    else :
                        output=0.6*output+0.4*vlm.get_image_features(vlm_data)
#                         output=vlm.forward(vlm_data)
                else:
                    output = net(data)
                
                # print(output.shape)                                     
                loss = criterion(output, target)

            optimizer.zero_grad()
            if self.use_fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            output = F.softmax(output, dim=1)
            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            acc = accuracy(output.data, target)[0]
            train_loss.update(loss.item(), data.size(0))
            train_acc.update(acc.item(), data.size(0))

            torch.cuda.empty_cache()

            print('epoch:{},step:{},train_loss:{:.5f},train_acc:{:.5f},lr:{}'.
                  format(epoch, step, loss.item(), acc.item(),
                         optimizer.param_groups[0]['lr']))

            if self.global_step % 10 == 0:
                self.writer.add_scalars('data/train_loss_acc', {
                    'train_loss': loss.item(),
                    'train_acc': acc.item()
                }, self.global_step)

            self.global_step += 1

        return train_loss.avg, train_acc.avg

    def _val_on_epoch(self, epoch, net, vlm,criterion, val_path, label_dict):

        net.eval()

        val_transformer = transforms.Compose([
            tr.ToCVImage(),
            tr.RandomResizedCrop(size=self.image_size, scale=(1.0, 1.0)),
            tr.ToTensor(),
          #  tr.Normalize(self.train_mean, self.train_std)
        ])

        val_dataset = DataGenerator(val_path,
                                    label_dict,
                                    transform=val_transformer)

        val_loader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers,
                                pin_memory=True)

        val_loss = AverageMeter()
        val_acc = AverageMeter()

        with torch.no_grad():
            for step, sample in enumerate(val_loader):
                data = sample['image']
                target = sample['label']

                data = data.cuda()
                target = target.cuda()
                
                with autocast(self.use_fp16):
                    output = net(data)
                    if self.vlm_trained:
                        vlm_data = sample['image']
                        if self.vlm_ft:
                             output=vlm.get_image_features(vlm_data)
#                             output=vlm.forward(vlm_data)
                        else:
                            output=0.6*output+0.4*vlm.get_image_features(vlm_data)
                       
                    loss = criterion(output, target)

                output = F.softmax(output, dim=1)
                output = output.float()
                loss = loss.float()

                # measure accuracy and record loss
                acc = accuracy(output.data, target)[0]
                val_loss.update(loss.item(), data.size(0))
                val_acc.update(acc.item(), data.size(0))

                torch.cuda.empty_cache()

                print('epoch:{},step:{},val_loss:{:.5f},val_acc:{:.5f}'.format(
                    epoch, step, loss.item(), acc.item()))

        return val_loss.avg, val_acc.avg

    def hook_fn_forward(self, module, input, output):

        for i in range(input[0].size(0)):
            self.feature_in.append(input[0][i].cpu().numpy())
            self.feature_out.append(output[i].cpu().numpy())

    def inference(self,
                  test_path,
                  label_dict,
                  net=None,
                  hook_fn_forward=False):

        if net is None:
            net = self.net
            
        if self.vlm_trained:
            vlm = VLMInterface(device=self.device)#创建vlm接口
        else:
            vlm=None

        if hook_fn_forward:
            net.avgpool.register_forward_hook(self.hook_fn_forward)

        net = net.cuda()
        net.eval()

        test_transformer = transforms.Compose([
            tr.ToCVImage(),
            tr.RandomResizedCrop(size=self.image_size, scale=(1.0, 1.0)),
            tr.ToTensor(),
            tr.Normalize(self.train_mean, self.train_std)
        ])

        test_dataset = DataGenerator(test_path,
                                     label_dict,
                                     transform=test_transformer)

        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 pin_memory=True)

        result = {'true': [], 'pred': [], 'prob': []}

        test_acc = AverageMeter()

        with torch.no_grad():
            for step, sample in enumerate(test_loader):
                data = sample['image']
                target = sample['label']

                data = data.cuda()
                target = target.cuda()  #N
                
                with autocast(self.use_fp16):
                    output = net(data)
                    if self.vlm_trained:
                        vlm_data = sample['image']
                        #print(vlm_data.shape)
                        output=0.6*output+0.4*vlm.get_image_features(vlm_data)
                   
                output = F.softmax(output, dim=1)
                output = output.float()  #N*C

                acc = accuracy(output.data, target)[0]
                test_acc.update(acc.item(), data.size(0))

                result['true'].extend(target.detach().tolist())
                result['pred'].extend(torch.argmax(output, 1).detach().tolist())
                result['prob'].extend(output.detach().tolist())

                print('step:{},test_acc:{:.5f}'.format(step, acc.item()))

                torch.cuda.empty_cache()

        print('average test_acc:{:.5f}'.format(test_acc.avg))

        return result, np.array(self.feature_in), np.array(self.feature_out)

    def _get_net(self, net_name):
            
            if net_name.startswith('res') or net_name.startswith('wide_res'):
                import model.resnet as resnet
                print("loading parameters……")
                net = resnet.__dict__[net_name](
                    pretrained=self.transfer_lr,  #修改
                    # type=self.transfer_type,		#修改
                    transfer_type=self.transfer_type,
                    num_classes=self.num_classes
                )
                

            elif net_name.startswith('vit_'):
                import model.vision_transformer as vit
                net = vit.__dict__[net_name](
                    #num_classes=self.num_classes,
                    image_size=self.image_size,
                    dropout=self.dropout
                )

            return net

    def _get_loss(self, loss_fun, class_weight=None):
        if class_weight is not None:
            class_weight = torch.tensor(class_weight)

        if loss_fun == 'Cross_Entropy':
            loss = nn.CrossEntropyLoss(class_weight)

        return loss

    def _get_optimizer(self, optimizer, net,vlm,lr):
        if self.vlm_trained and self.vlm_ft:
            para=vlm.model.parameters()
            optimizer = torch.optim.Adam(vlm.model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-6 ,weight_decay=self.weight_decay) 
            # optimizer = torch.optim.Adam(vlm.model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6 ,weight_decay=0.2) 
        elif self.vlm_trained:
            para=net.parameters()+vlm.model.parameters()
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

    def _get_lr_scheduler(self, lr_scheduler, optimizer):
            if lr_scheduler == 'ReduceLROnPlateau':
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', patience=5, verbose=True)
            elif lr_scheduler == 'MultiStepLR':
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, self.milestones, gamma=self.gamma)
            elif lr_scheduler == 'CosineAnnealingLR':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.T_max)
            elif lr_scheduler == 'CosineAnnealingWarmRestarts':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, 20, T_mult=2)
            elif lr_scheduler == 'ExponentialLR':
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98, last_epoch=-1)

            return lr_scheduler

    def _get_pre_trained(self, weight_path):
        checkpoint = torch.load(weight_path)
        if self.vlm_trained and self.vlm_ft:
            self.vlm.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.net.load_state_dict(checkpoint['state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1


# computing tools


class AverageMeter(object):
    '''
    Computes and stores the average and current value
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    '''
    Computes the precision@k for the specified values of k
    '''
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res


class EarlyStopping(object):
    """Early stops the training if performance doesn't improve after a given patience."""
    def __init__(self,
                 patience=10,
                 verbose=True,
                 delta=0,
                 monitor='val_loss',
                 best_score=None,
                 op_type='min'):
        """
        Args:
            patience (int): How long to wait after last time performance improved.
                            Default: 10
            verbose (bool): If True, prints a message for each performance improvement. 
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            monitor (str): Monitored variable.
                            Default: 'val_loss'
            op_type (str): 'min' or 'max'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.delta = delta
        self.monitor = monitor
        self.op_type = op_type

        if self.op_type == 'min':
            self.val_score_min = np.Inf
        else:
            self.val_score_min = 0

    def __call__(self, val_score):

        score = -val_score if self.op_type == 'min' else val_score

        if self.best_score is None:
            self.best_score = score
            self.print_and_update(val_score)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}'
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.print_and_update(val_score)
            self.counter = 0

    def print_and_update(self, val_score):
        '''print_message when validation score decrease.'''
        if self.verbose:
            print(
                self.monitor,
                f'optimized ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ...'
            )
        self.val_score_min = val_score