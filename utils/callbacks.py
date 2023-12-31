import os

import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal

import cv2
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .utils import cvtColor, preprocess_input, resize_image
from .utils_metrics import compute_mIoU


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        try:
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")


class EvalCallback():
    def __init__(self, net, input_shape, num_classes, image_ids, mask_ids, log_dir, cuda,
                 miou_out_path="./temp_miou_out", eval_flag=True, period=1):
        super(EvalCallback, self).__init__()

        self.net = net
        self.input_shape = input_shape
        self.num_classes = num_classes
        # self.image_ids = image_ids
        # self.mask_ids = mask_ids
        # self.dataset_path = dataset_path
        self.log_dir = log_dir
        self.cuda = cuda
        self.miou_out_path = miou_out_path
        self.eval_flag = eval_flag
        self.period = period

        self.image_ids = [image_id.split()[0] for image_id in image_ids]
        self.mask_ids = [mask_id.split()[0] for mask_id in mask_ids]
        self.mious = [0]
        self.epoches = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def get_miou_png(self, image_list):
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        images = []
        for image_dir in image_list:
            image = Image.open(image_dir)
            image = cvtColor(image)
            original_h = np.array(image).shape[0]
            original_w = np.array(image).shape[1]
            # ---------------------------------------------------------#
            #   给图像增加灰条，实现不失真的resize
            #   也可以直接resize进行识别
            # ---------------------------------------------------------#
            image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
            # ---------------------------------------------------------#
            #   添加上batch_size维度
            # ---------------------------------------------------------#
            # image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
            image_data = np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1))
            images.append({
                'image': torch.from_numpy(image_data).to("cuda:0"),
                'original_size': (original_h, original_w)
                })

        from segment_anything import SamAutomaticMaskGenerator
        mask_generator = SamAutomaticMaskGenerator(
            model=self.net.module,
            points_per_batch=128,
            box_nms_thresh=0.1,
            crop_n_layers=1,
            crop_nms_thresh=0.5,
            crop_n_points_downscale_factor=2,
        )

        with torch.no_grad():
            # images = torch.from_numpy(image_data).float()
            # if self.cuda:
            #     images = images.cuda()

            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            # pr = []
            # for img in images:
            #     masks = mask_generator.generate(img.permute(1, 2, 0))
            #     mask = torch.zeros((masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1]))
            #     for i in range(len(masks)):
            #         mask = torch.logical_or(mask, torch.from_numpy(masks[i]['segmentation']))
            #     pr.append(mask.float())
            # pr = torch.stack(pr)
            batched_output = self.net.module(images, multimask_output=False)
            prs = [output['masks'].float().squeeze(0) for output in batched_output]
            # pr = torch.cat(pr, dim=0).squeeze(1)
            img_out = []
            for i in range(len(prs)):
                pr = prs[i]
                # ---------------------------------------------------#
                #   取出每一个像素点的种类
                # ---------------------------------------------------#
                pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
                # --------------------------------------#
                #   将灰条部分截取掉
                # --------------------------------------#
                pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
                # ---------------------------------------------------#
                #   进行图片的resize
                # ---------------------------------------------------#
                pr = np.expand_dims(cv2.resize(pr, (images[i]['original_size'][1], images[i]['original_size'][0]), interpolation=cv2.INTER_LINEAR), axis=-1)
                # ---------------------------------------------------#
                #   取出每一个像素点的种类
                # ---------------------------------------------------#
                pr = pr.argmax(axis=-1)

                image = Image.fromarray(np.uint8(pr))
                img_out.append(image)
        return img_out

    def on_epoch_end(self, epoch, model_eval):
        batch_size = 8
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            # gt_dir = os.path.join(self.dataset_path, "VOC2007/SegmentationClass/")
            gt_dir = self.image_ids.copy()
            for i in range(len(gt_dir)):
                gt_dir[i] = gt_dir[i][:-3] + 'png'
                parts = gt_dir[i].split('/')
                parts[3] = '0'
                parts[4] = 'label_data_pngs_aug'
                gt_dir[i] = os.path.join(*parts)
            # pred_dir_part = os.path.join(self.miou_out_path, 'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
                # os.makedirs(self.miou_out_path + '/2019')
                # os.makedirs(self.miou_out_path + '/2020')
                # os.makedirs(self.miou_out_path + '/2021')
                # os.makedirs(self.miou_out_path + '/2022')
                os.makedirs(f"{self.miou_out_path}/0/label_data_pngs_aug")
            # if not os.path.exists(pred_dir_part):
            #     os.makedirs(pred_dir_part)
            print("Get miou.")
            pred_dir = []
            for i in tqdm(range(0, len(self.image_ids), batch_size)):
                # -------------------------------#
                #   从文件中读取图像
                # -------------------------------#
                # image_path = os.path.join(self.dataset_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
                # image = Image.open(image_path)
                image_id_list = self.image_ids[i:(i+batch_size)]
                # ------------------------------#
                #   获得预测txt
                # ------------------------------#
                images = self.get_miou_png(image_id_list)
                for j in range(len(images)):
                    image_id = image_id_list[j]
                    image = images[j]
                    # save_dir = self.miou_out_path+'/'+image_id[12:16]+image_id[23:-4]+".png"
                    save_dir = f"{self.miou_out_path}/0/label_data_pngs_aug/{image_id[57:-4]}.png"
                    image.save(save_dir)
                    pred_dir.append(save_dir)

            print("Calculate miou.")
            # png_list = self.image_ids.copy()
            # for i in range(len(png_list)):
            #     png_list[i] = png_list[i][24:-4]
            _, IoUs, _, _ = compute_mIoU(gt_dir, pred_dir, self.num_classes, None)  # 执行计算mIoU的函数
            temp_miou = np.nanmean(IoUs) * 100

            self.mious.append(temp_miou)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(temp_miou))
                f.write("\n")

            plt.figure()
            plt.plot(self.epoches, self.mious, 'red', linewidth=2, label='train miou')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou')
            plt.title('A Miou Curve')
            plt.legend(loc="lower right")

            plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
            plt.cla()
            plt.close("all")

            print("Get miou done.")
            shutil.rmtree(self.miou_out_path)
