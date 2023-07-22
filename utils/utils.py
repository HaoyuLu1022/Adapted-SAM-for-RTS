import numpy as np
from PIL import Image
import torch
import random

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh
    
#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def preprocess_input(image):
    image /= 255.0
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def show_configs(kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'mobilenet' : 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar',
        'xception'  : 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth',
    }
    url = download_urls[backbone]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)

# from https://github.com/WuJunde/Medical-SAM-Adapter/blob/main/utils.py
def generate_click_prompt(img, msk, pt_label = 1):
    # return: prompt, prompt mask
    pt_list = []
    msk_list = []
    msk = msk.unsqueeze(1)
    b, c, h, w = msk.size()
    msk = msk[:,0,:,:]
    
    pt_list_s = []
    msk_list_s = []
    for j in range(b):
        msk_s = msk[j,:,:]
        indices = torch.nonzero(msk_s)
        if indices.size(0) == 0:
            # generate a random array between [0-h, 0-h]:
            random_index = torch.randint(0, h, (2,)).to(device = msk.device)
            new_s = msk_s
        else:
            random_index = random.choice(indices)
            label = msk_s[random_index[0], random_index[1]]
            new_s = torch.zeros_like(msk_s)
            # convert bool tensor to int
            new_s = (msk_s == label).to(dtype = torch.float)
            # new_s[msk_s == label] = 1
        pt_list_s.append(random_index)
        msk_list_s.append(new_s)
    pts = torch.stack(pt_list_s, dim=0)
    msks = torch.stack(msk_list_s, dim=0)
    pt_list.append(pts)
    msk_list.append(msks)
    
    pt = torch.stack(pt_list, dim=-1).squeeze(-1)
    msk = torch.stack(msk_list, dim=-1).squeeze(-1)

    # msk = msk.unsqueeze(1)

    return img, pt, msk #[b, 2], [b, c, h, w]

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#---------------------------------------------------#
#   设置Dataloader的种子
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

