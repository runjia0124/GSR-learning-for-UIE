import logging
import torch
from PIL import Image
from torchvision import transforms
from unet.UNet_new import Unet_resize_conv
from util.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from unet.ANet.attention_net import Attention_net
from options.test_options import TestOptions
import numpy as np

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def get_config(config):
    import yaml
    with open(config, 'r') as stream:
        return yaml.load(stream)

def norm(w):
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            w[i][j] = -1 + 2 * (w[i][j] - 0)
    return w

skip = False
opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
dir_img = opt.dataroot
dir_mask = dir_img

netG_A = Unet_resize_conv(opt, skip)
netG_B = Unet_resize_conv(opt, skip)
net_addition = Attention_net(opt, skip)
logging.info("Loading model {}".format(opt.model))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device {device}')
netG_A.to(device=device)
netG_B.to(device=device)
net_addition.to(device=device)
netG_A.load_state_dict(torch.load('./checkpoints/new/CP_A_epoch2_1237.pth', map_location=device))
netG_B.load_state_dict(torch.load('./checkpoints/new/CP_B_epoch2_1237.pth', map_location=device))
net_addition.load_state_dict(torch.load('./checkpoints/new/CP_Attention_epoch2_1237.pth', map_location=device))

dataset = BasicDataset(dir_img, dir_mask)
dataset_size = len(dataset)
test_loader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=0, pin_memory=True)
print('#testing images = %d' % dataset_size)

# test length
print(len(dataset))

with torch.no_grad():
    for i, data in enumerate(test_loader):

        img = data['image']
        img = img.to(device=device, dtype=torch.float32)
        path = data['name']

        A = netG_A(img)
        m = netG_B(img)
        w_1 = net_addition(img)[0][0]
        w_2 = net_addition(img)[0][1]
        G = m * img
        L = A * (1 - m)

        img = G * w_1 + L * w_2
        img = transforms.ToPILImage()(img[0].cpu()).convert('RGB')
        img.save("./datasets/results/{}.png".format(path[0]))



