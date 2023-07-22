from train_function import train
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4196"

if __name__ == '__main__':
    ver = "sam_vit_b_01ec64"
    model_path = f"model_weights/{ver}.pth"
    train(model_path, ver)

