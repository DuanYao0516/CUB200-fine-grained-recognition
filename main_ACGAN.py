# import os
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.utils import save_image
# import numpy as np

# from torch.nn import functional as F

# import data_utils.transform as tr
# from data_utils.data_loader import DataGenerator

# from model.acgan import ACGANGenerator, ACGANDiscriminator  # 导入ACGAN模型

# from config import CUB_TRAIN_MEAN, CUB_TRAIN_STD
# from data_utils.csv_reader import csv_reader_single

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def acgan_trainer(image_size, encoding_dims, num_classes, batch_size, epochs, num_workers):
#     np.random.seed(0)
#     torch.cuda.manual_seed_all(0)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = True

#     generator = ACGANGenerator(num_classes=num_classes, encoding_dims=encoding_dims, out_size=image_size, out_channels=3)
#     discriminator = ACGANDiscriminator(num_classes=num_classes, in_size=image_size, in_channels=3)

#     csv_path = './csv_file/cub_200_2011.csv_train.csv'
#     label_dict = csv_reader_single(csv_path, key_col='id', value_col='label')
#     train_path = list(label_dict.keys())

#     train_transformer = transforms.Compose([
#         tr.ToCVImage(),
#         tr.RandomResizedCrop(image_size),
#         tr.ToTensor(),
#         tr.Normalize(CUB_TRAIN_MEAN, CUB_TRAIN_STD)
#     ])

#     train_dataset = DataGenerator(train_path,
#                                   label_dict,
#                                   transform=train_transformer)

#     train_loader = DataLoader(train_dataset,
#                               batch_size=batch_size,
#                               shuffle=True,
#                               num_workers=num_workers,
#                               pin_memory=True)

#     generator = generator.cuda()
#     discriminator = discriminator.cuda()

#     optimG = torch.optim.AdamW(generator.parameters(), 0.0002, betas=(0.5, 0.999))
#     optimD = torch.optim.AdamW(discriminator.parameters(), 0.0002, betas=(0.5, 0.999))

#     loss = nn.BCELoss()

#     for epoch in range(1, epochs+1):
#         for step, sample in enumerate(train_loader, 0):

#             images = sample['image'].to(device)
#             labels = sample['label'].to(device)  # 获取训练集中的真实标签

#             bs = images.size(0)

#             # ---------------------
#             #       disc
#             # ---------------------
#             optimD.zero_grad()

#             # real
#             # pvalidity_real, class_pred_real = discriminator(images, mode="classifier")
#             # pvalidity_real = F.sigmoid(pvalidity_real)
#             # errD_real = loss(pvalidity_real, torch.full((bs,), 1.0, device=device))
#             # errD_real.backward()
#             pvalidity_real, class_pred_real = discriminator(images, mode="classifier")
#             errD_real = loss(pvalidity_real, torch.full((bs,), 1.0, device=device))
#             errD_real.backward()

#             # fake
#             # noise = torch.randn(bs, encoding_dims, 1, 1, device=device)
#             # fake_labels = torch.randint(0, num_classes, (bs,), device=device)
#             # fakes = generator(noise, fake_labels)
#             # pvalidity_fake, class_pred_fake = discriminator(fakes.detach(), mode="classifier")
#             # pvalidity_fake = F.sigmoid(pvalidity_fake)
#             # errD_fake = loss(pvalidity_fake, torch.full((bs,), 0.0, device=device))
#             # errD_fake.backward()
#             noise = torch.randn(bs, encoding_dims, 1, 1, device=device)
#             fake_labels = torch.randint(0, num_classes, (bs,), device=device)
#             fakes = generator(noise, fake_labels)
#             pvalidity_fake, class_pred_fake = discriminator(fakes.detach(), mode="classifier")
#             errD_fake = loss(pvalidity_fake, torch.full((bs,), 0.0, device=device))
#             errD_fake.backward()

#             # update disc params
#             optimD.step()

#             # ---------------------
#             #        gen
#             # ---------------------
#             optimG.zero_grad()

#             noise = torch.randn(bs, encoding_dims, 1, 1, device=device)
#             gen_labels = torch.randint(0, num_classes, (bs,), device=device)
#             gen_images = generator(noise, gen_labels)
#             # pvalidity_gen, class_pred_gen = discriminator(gen_images)
#             pvalidity, class_pred = discriminator(gen_images, mode="classifier")
#             # pvalidity_gen = F.sigmoid(pvalidity_gen)
#             # errG = loss(pvalidity_gen, torch.full((bs,), 1.0, device=device))
#             # errG.backward()
#             errG = loss(pvalidity, torch.full((bs,), 1.0, device=device))
#             errG.backward()

#             optimG.step()

#             print("[{}/{}] [{}/{}] G_loss: [{:.4f}] D_loss_real: [{:.4f}] D_loss_fake: [{:.4f}]"
#                   .format(epoch, epochs, step, len(train_loader), errG, errD_real, errD_fake))

#     torch.save(generator.state_dict(), 'ckpt/acgan_generator.pth')
#     torch.save(discriminator.state_dict(), 'ckpt/acgan_discriminator.pth')

    
# if __name__ == "__main__":
#     image_size = 256
#     encoding_dims = 100
#     num_classes = 200  # 根据你的训练集中类别的数量设置
#     batch_size = 100
#     num_workers = 10
#     epochs = 300
#     number_gen = 10

#     if os.path.exists('ckpt/acgan_generator.pth'):
#         if not os.path.exists('gen_dataset/'):
#             os.makedirs('gen_dataset/')

#         generator = ACGANGenerator(num_classes=num_classes, encoding_dims=encoding_dims, out_size=image_size, out_channels=3)
#         generator = generator.cuda()
#         checkpoint = torch.load('ckpt/acgan_generator.pth')
#         generator.load_state_dict(checkpoint)

#         noise = torch.randn(number_gen, encoding_dims, 1, 1, device=device)
#         gen_labels = torch.randint(0, num_classes, (number_gen,), device=device)  # 使用随机生成的标签
#         gen_images = generator(noise, gen_labels).detach()
        
#         # 指定要生成的类别（假设生成第一个类别的图像）
#         class_to_generate = 0
        
#         for i in range(number_gen):
#             noise = torch.randn(1, encoding_dims, 1, 1, device=device)  # 生成单个样本
#             gen_labels = torch.tensor([class_to_generate], device=device)  # 使用指定的类别标签
#             gen_image = generator(noise, gen_labels).detach().squeeze(0)
#             save_image(gen_image, 'gen_dataset/{}_{}.jpg'.format(class_to_generate, i))

#         # for i in range(number_gen):

#     else:
#         acgan_trainer(image_size, encoding_dims, num_classes, batch_size, epochs, num_workers)


import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np

import data_utils.transform as tr
from data_utils.data_loader import DataGenerator

from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

# Import ACGAN model
from model.acgan import ACGANGenerator, ACGANDiscriminator 

from config import CUB_TRAIN_MEAN, CUB_TRAIN_STD
from data_utils.csv_reader import csv_reader_single

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def acgan_trainer(image_size, encoding_dims, num_classes, batch_size, epochs, num_workers):

    np.random.seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    generator = ACGANGenerator(encoding_dims=encoding_dims, num_classes=num_classes, out_size=image_size, out_channels=3)
    discriminator = ACGANDiscriminator(in_size=image_size, in_channels=3, num_classes=num_classes)

    csv_path = './csv_file/cub_200_2011.csv_train.csv'
    label_dict = csv_reader_single(csv_path, key_col='id', value_col='label')
    train_path = list(label_dict.keys())
    
    train_transformer = transforms.Compose([
        tr.ToCVImage(),
        tr.RandomResizedCrop(image_size),
        tr.ToTensor(),
        tr.Normalize(CUB_TRAIN_MEAN, CUB_TRAIN_STD)
    ])
    
    train_dataset = DataGenerator(train_path,
                                  label_dict,
                                  transform=train_transformer)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)
    
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    
    optimG = torch.optim.AdamW(generator.parameters(), 0.0002, betas=(0.5, 0.999))
    optimD = torch.optim.AdamW(discriminator.parameters(), 0.0002, betas=(0.5, 0.999))
    
    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    for epoch in range(1, epochs+1):
        for step, sample in enumerate(train_loader, 0):
            
            images = sample['image'].to(device)
            labels = sample['label'].to(device)
            bs = images.size(0)
            
            # ---------------------
            #     Discriminator
            # ---------------------
            optimD.zero_grad()
        
            # Real images
            real_validity = discriminator(images, mode="discriminator")
            real_validity = torch.sigmoid(real_validity)
            errD_real = bce_loss(real_validity, torch.ones(bs, device=device))
            
            class_pred_real = discriminator(images, mode="classifier")
            errD_real_class = ce_loss(class_pred_real, labels)
            
            errD_real_total = errD_real + errD_real_class
            errD_real_total.backward()
            
            # Fake images
            noise = torch.randn(bs, encoding_dims, device=device)
            gen_labels = torch.randint(0, num_classes, (bs,), device=device)
            fakes = generator(noise, gen_labels)
            
            fake_validity = discriminator(fakes.detach(), mode="discriminator")
            fake_validity = torch.sigmoid(fake_validity)
            errD_fake = bce_loss(fake_validity, torch.zeros(bs, device=device))
            
            errD_fake.backward()
        
            # Update discriminator
            optimD.step()
            optimD.step()
        
            # ------------------------
            #      Generator
            # ------------------------
            optimG.zero_grad()
        
            gen_validity = discriminator(fakes, mode="discriminator")
            gen_validity = torch.sigmoid(gen_validity)
            errG = bce_loss(gen_validity, torch.ones(bs, device=device))
            
            class_pred_fake = discriminator(fakes, mode="classifier")
            errG_class = ce_loss(class_pred_fake, gen_labels)
            
            errG_total = errG + errG_class
            errG_total.backward()
        
            # Update generator
            optimG.step()
        
            print("[{}/{}] [{}/{}] G_loss: [{:.4f}] D_loss: [{:.4f}]"
                  .format(epoch, epochs, step, len(train_loader), errG_total, errD_real_total + errD_fake))
            
    torch.save(generator.state_dict(), 'ckpt/acgan_generator.pth')
    torch.save(discriminator.state_dict(), 'ckpt/acgan_discriminator.pth')

if __name__ == "__main__":
    image_size = 256
    encoding_dims = 100
    num_classes = 200
    batch_size = 100
    num_workers = 10
    epochs = 300
    number_gen = 10
    
    if os.path.exists('ckpt/acgan_generator.pth'):
        if not os.path.exists('gen_dataset/'):
            os.makedirs('gen_dataset/')
            
        generator = ACGANGenerator(encoding_dims=encoding_dims, num_classes=num_classes, out_size=image_size, out_channels=3)
        generator = generator.cuda()
        checkpoint = torch.load('ckpt/acgan_generator.pth')
        generator.load_state_dict(checkpoint)
        
        noise = torch.randn(number_gen, encoding_dims, device=device)
        gen_labels = torch.randint(0, num_classes, (number_gen,), device=device)
        gen_images = generator(noise, gen_labels).detach()
        
        for i in range(number_gen):
            save_image(gen_images[i], 'gen_dataset/'+str(i)+'.jpg')

    else:
        acgan_trainer(image_size, encoding_dims, num_classes, batch_size, epochs, num_workers)
