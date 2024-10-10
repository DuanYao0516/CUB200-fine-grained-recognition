import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt

from torch.nn import functional as F

import data_utils.transform as tr
from data_utils.data_loader import DataGenerator

from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

import model.jq_dcgan as dcgan
from config import CUB_TRAIN_MEAN, CUB_TRAIN_STD
from data_utils.csv_reader import csv_reader_single

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def gan_trainer(image_size, encoding_dims, batch_size, epochs, num_workers):
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    generator = dcgan.Generator(ngpu=1)
    discriminator = dcgan.Discriminator(ngpu=1)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    csv_path = './csv_file/cub_200_2011.csv_train.csv'
    label_dict = csv_reader_single(csv_path, key_col='id', value_col='label')
    train_path = list(label_dict.keys())
    
    train_transformer = transforms.Compose([
        tr.ToCVImage(),
        tr.RandomResizedCrop(image_size),
        tr.ToTensor(),
        tr.Normalize(CUB_TRAIN_MEAN, CUB_TRAIN_STD)
    ])
    
    train_dataset = DataGenerator(train_path, label_dict, transform=train_transformer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    optimG = torch.optim.AdamW(generator.parameters(), 0.0001, betas=(0.5, 0.999))
    optimD = torch.optim.AdamW(discriminator.parameters(), 0.0001, betas=(0.5, 0.999))
    
    loss = nn.BCELoss()
    
    G_losses = []
    D_losses = []
    
    for epoch in range(1, epochs + 1):
        G_loss_epoch = 0.0
        D_loss_epoch = 0.0
        for step, sample in enumerate(train_loader, 0):
            images = sample['image'].to(device)
            bs = images.size(0)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimD.zero_grad()
            
            # Train with real images
            pvalidity_real = discriminator(images)
            errD_real = loss(pvalidity_real, torch.full((bs, 1), 0.9, device=device))  # label smoothing
            errD_real.backward()
            
            # Train with fake images
            noise = torch.randn(bs, encoding_dims, 1, 1, device=device)
            fakes = generator(noise)
            pvalidity_fake = discriminator(fakes.detach())
            errD_fake = loss(pvalidity_fake, torch.full((bs, 1), 0.1, device=device))  # label smoothing
            errD_fake.backward()
            
            errD = errD_real + errD_fake
            optimD.step()
        
            # ---------------------
            #  Train Generator
            # ---------------------
            optimG.zero_grad()
            
            pvalidity_fake = discriminator(fakes)
            errG = loss(pvalidity_fake, torch.full((bs, 1), 0.9, device=device))  # label smoothing
            errG.backward()
            
            optimG.step()
            
            G_loss_epoch += errG.item()
            D_loss_epoch += errD.item()
            
            if step % 10 == 0:
                print("[{}/{}] [{}/{}] G_loss: [{:.4f}] D_loss: [{:.4f}]".format(epoch, epochs, step, len(train_loader), errG.item(), errD.item()))
        
        G_losses.append(G_loss_epoch / len(train_loader))
        D_losses.append(D_loss_epoch / len(train_loader))
    
    torch.save(generator.state_dict(), 'ckpt/generator.pth')
    torch.save(discriminator.state_dict(), 'ckpt/discriminator.pth')
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), G_losses, 'g-', label='G_loss')
    plt.plot(range(1, epochs + 1), D_losses, 'b-', label='D_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Generator and Discriminator Loss Over Epochs')
    plt.savefig('loss_curve.png')
    plt.show()

if __name__ == "__main__":
    image_size = 256
    encoding_dims = 100
    batch_size = 100
    num_workers = 10
    epochs = 2000
    number_gen = 10
    
    if os.path.exists('ckpt/generator.pth'):
        if not os.path.exists('gen_dataset/'):
            os.makedirs('gen_dataset/')
            
        generator = dcgan.Generator(ngpu=1)
        generator = generator.to(device)
        checkpoint = torch.load('ckpt/generator.pth')
        generator.load_state_dict(checkpoint)
        
        noise = torch.randn(number_gen, encoding_dims, 1, 1, device=device)
        gen_images = generator(noise).detach()
        
        for i in range(number_gen):
            save_image(gen_images[i], 'gen_dataset/' + str(i) + '.jpg')
    else:
        gan_trainer(image_size, encoding_dims, batch_size, epochs, num_workers)
