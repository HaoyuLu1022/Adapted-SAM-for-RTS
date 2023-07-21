from train_function import train
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4196"

if __name__ == '__main__':
    model_path = "model_weights/sam_vit_h_4b8939.pth"
    ver = "vit_h"
    train(model_path, ver)

