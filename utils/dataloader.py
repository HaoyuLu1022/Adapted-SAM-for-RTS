import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class SAMAdaDataset(Dataset):
    def __init__(self, input_shape, output_shape, num_classes, train, dataset_path, annotation_lines=None):
        super(SAMAdaDataset, self).__init__()
        self.annotation_lines = annotation_lines
        # self.length             = len(annotation_lines)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path

        self.img_path = os.path.join(self.dataset_path, 'images')
        self.img_list = os.listdir(self.img_path)
        self.mask_path = os.path.join(self.dataset_path, 'masks')
        self.mask_list = os.listdir(self.mask_path)
        self.label_path = os.path.join(self.dataset_path, 'labels')
        self.label_list = os.listdir(self.label_path)

    # 考虑到source和target数据集可能大小不同，暂时与两者中最小值对齐
    #  并通过align_dataset成员判断与两者哪一个对其
    def __len__(self):
        self.length = len(self.img_list)
        return self.length

    def __getitem__(self, index):
        # annotation_line = self.annotation_lines[index]
        # name = annotation_line.split()[0]
    
        #-------------------------------#
        #   从文件中读取图像
        #-------------------------------#
        jpg         = Image.open(os.path.join(self.img_path, self.img_list[index]))
        png         = Image.open(os.path.join(self.mask_path, self.mask_list[index]))
        #-------------------------------#
        #   数据增强
        #-------------------------------#
        jpg, png    = self.get_random_data(jpg, png, self.input_shape, self.output_shape, random = self.train)

        jpg         = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2,0,1])
        png         = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        #-------------------------------------------------------#
        #   转化成one_hot的形式
        #   在这里需要+1是因为voc数据集有些标签具有白边部分
        #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
        #-------------------------------------------------------#
        seg_labels  = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels  = seg_labels.reshape((int(self.output_shape[0]), int(self.output_shape[1]), self.num_classes + 1))

        dir = os.path.join(self.img_path, self.img_list[index])

        return jpg, png, seg_labels, dir

    def get_img_list(self):
        return self.img_list

    def rand(self, a: float = 0, b: float = 1) -> float:
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, output_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        iw, ih = image.size
        h_in, w_in = input_shape
        h_out, w_out = output_shape

        if not random:
            iw, ih = image.size
            scale_in = min(w_in / iw, h_in / ih)
            nw_in = int(iw * scale_in)
            nh_in = int(ih * scale_in)

            ow, oh = label.size
            scale_out = min(w_out / ow, h_out / oh)
            nw_out = int(ow * scale_out)
            nh_out = int(oh * scale_out)

            image = image.resize((nw_in, nh_in), Image.BICUBIC)
            new_image = Image.new('RGB', (w_in, h_in), (128, 128, 128))
            new_image.paste(image, ((w_in - nw_in) // 2, (h_in - nh_in) // 2))

            label = label.resize((nw_out, nh_out), Image.NEAREST)
            new_label = Image.new('L', (w_out, h_out), (0))
            new_label.paste(label, ((w_out - nw_out) // 2, (h_out - nh_out) // 2))
            return new_image, new_label

        # ------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        # ------------------------------------------#
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh_in = int(scale * h_in)
            nw_in = int(nh_in * new_ar)
            nh_out = int(scale * h_out)
            nw_out = int(nh_out * new_ar)
        else:
            nw_in = int(scale * w_in)
            nh_in = int(nw_in/ new_ar)
            nw_out = int(scale * w_out)
            nh_out = int(nw_out/ new_ar)
        image = image.resize((nw_in, nh_in), Image.BICUBIC)
        label = label.resize((nw_out, nh_out), Image.NEAREST)

        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # ------------------------------------------#
        #   将图像多余的部分加上灰条
        # ------------------------------------------#
        dx_in = int(self.rand(0, w_in - nw_in))
        dy_in = int(self.rand(0, h_in - nh_in))
        dx_out = int(self.rand(0, w_out - nw_out))
        dy_out = int(self.rand(0, h_out - nh_out))
        new_image = Image.new('RGB', (w_in, h_in), (128, 128, 128))
        new_label = Image.new('L', (w_out, h_out), (0))
        new_image.paste(image, (dx_in, dy_in))
        new_label.paste(label, (dx_out, dy_out))
        image = new_image
        label = new_label

        image_data = np.array(image, np.uint8)

        # ------------------------------------------#
        #   高斯模糊
        # ------------------------------------------#
        blur = self.rand() < 0.25
        if blur:
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        # ------------------------------------------#
        #   旋转
        # ------------------------------------------#
        rotate = self.rand() < 0.25
        if rotate:
            center_in = (w_in // 2, h_in // 2)
            center_out = (w_out // 2, h_out // 2)
            rotation = np.random.randint(-10, 11)
            M_in = cv2.getRotationMatrix2D(center_in, -rotation, scale=1)
            M_out = cv2.getRotationMatrix2D(center_out, -rotation, scale=1)
            image_data = cv2.warpAffine(image_data, M_in, (w_in, h_in), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128))
            label = cv2.warpAffine(np.array(label, np.uint8), M_out, (w_out, h_out), flags=cv2.INTER_NEAREST, borderValue=(0))

        # ---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        # ---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # ---------------------------------#
        #   将图像转到HSV上
        # ---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        # ---------------------------------#
        #   应用变换
        # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data, label


# DataLoader中collate_fn使用
# 为了实现batch前半和后半分离，实际的batch会是参数设置的两倍
def deeplab_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    for img, png, labels, dir in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels

