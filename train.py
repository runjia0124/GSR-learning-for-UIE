import time
import os
import torch
import logging
from tqdm import tqdm
from torch import optim
from options.train_options import TrainOptions
from torch.utils.tensorboard import SummaryWriter
from models import networks
import torch.nn as nn
from util.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from unet.UNet_new import Unet_resize_conv
from unet.ANet.attention_net import Attention_net
def get_config(config):
    import yaml
    with open(config, 'r') as stream:
        return yaml.load(stream)

dir_img = "./datasets/trainA/"
dir_mask = "./datasets/trainB/"
dir_checkpoint = "./checkpoints/"

opt = TrainOptions().parse() # print options
skip = False

config = get_config(opt.config)
dataset = BasicDataset(dir_img, dir_mask)
dataset_size = len(dataset)
train_loader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=0, pin_memory=True)
print('#training images = %d' % dataset_size)

netG_A = Unet_resize_conv(opt, skip)
print('---------- Networks_A initialized -------------')
networks.print_network(netG_A)

netG_B = Unet_resize_conv(opt, skip)
print('---------- Networks_B initialized -------------')
networks.print_network(netG_B)

net_Attention = Attention_net(opt, skip)
print('---------- net_Attention initialized -------------')
networks.print_network(net_Attention)

global_step = 0

writer = SummaryWriter(comment=f'LR_{opt.lr}_BS_{opt.batchSize}')

optimizer_A = optim.RMSprop(netG_A.parameters(), lr=opt.lr, weight_decay=1e-8, momentum=0.9)
optimizer_B = optim.RMSprop(netG_B.parameters(), lr=opt.lr, weight_decay=1e-8, momentum=0.9)
optimizer_Attention = optim.RMSprop(net_Attention.parameters(), lr=opt.lr, weight_decay=1e-8, momentum=0.9)

criterion = nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.info(f'Using device {device}')

for epoch in range(1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    with tqdm(total=dataset_size, desc=f'Epoch {epoch}/{opt.niter + opt.niter_decay + 1}', unit='img') as pbar:
        for batch in train_loader:
            imgs = batch['image']
            reference = batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)

            netG_A = netG_A.to(device=device, dtype=torch.float32)
            netG_B = netG_B.to(device=device, dtype=torch.float32)
            net_Attention = net_Attention.to(device=device, dtype=torch.float32)

            mask_type = torch.float32
            reference = reference.to(device=device, dtype=mask_type)

            A = netG_A(imgs)
            m = netG_B(imgs)

            w_1 = net_Attention(imgs)[0][0]
            w_2 = net_Attention(imgs)[0][1]
            # transformations
            glb = A - A * m
            lc = m * imgs
            pred = glb * w_1 + lc * w_2

            loss = criterion(pred, reference)
            epoch_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), global_step)
            pbar.set_postfix(**{'loss (batch)': loss.item()})

            optimizer_A.zero_grad()
            optimizer_B.zero_grad()
            optimizer_Attention.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(netG_A.parameters(), 0.1)
            nn.utils.clip_grad_value_(netG_B.parameters(), 0.1)
            nn.utils.clip_grad_value_(net_Attention.parameters(), 0.1)
            optimizer_A.step()
            optimizer_B.step()
            optimizer_Attention.step()

            pbar.update(imgs.shape[0])
            global_step += 1
            if global_step % (len(dataset) // (10 * opt.batchSize)) == 0:
                for tag, value in netG_A.named_parameters():
                    tag = tag.replace('.', '/')
                    writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                for tag, value in netG_B.named_parameters():
                    tag = tag.replace('.', '/')
                    writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                writer.add_images('images', imgs, global_step)

        if True:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(netG_A.state_dict(),
                       dir_checkpoint + f'CP_A_epoch{epoch + 1}.pth')
            torch.save(netG_B.state_dict(),
                       dir_checkpoint + f'CP_B_epoch{epoch + 1}.pth')
            torch.save(net_Attention.state_dict(),
                       dir_checkpoint + f'CP_Attention_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()

torch.save(netG_A.state_dict(), "model_A.pth")
torch.save(netG_B.state_dict(), "model_B.pth")
torch.save(net_Attention.state_dict(), "Attention.pth")
