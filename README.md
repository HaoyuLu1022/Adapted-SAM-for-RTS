# Adapted-SAM-for-RTS

## Introduction

Implementing SAM (Segment Anything) with Adapter for RTS segementation. The model is kind of like finetuning over the original SAM with much less cost on parameters.

## Quick Start

Clone this repository, create a new folder named "model_weights" inside it, where the pretrained SAM models should be put. 

```sh
cd Adapted-SAM-for-RTS
mkdir model_weights
```

Then download the pretrained models inside the `model_weights` folder.

```sh
cd model_weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

Install necessary packages, run `main.py` for training, and you are good to go. Note that `segment_anything` package from `pip` is not neccessary since you are using a modified version of SAM.

```sh
python main.py
```

## Logs

- 23-07-19: Now you can train this model. The input should be 1024x1024x3 images, and the model outputs masks of 256x256x1.
  - **Bugs**: `eval_callback` is currently not available, so leave `eval_flag` to `False` for now. 
  - Note that for one single NVIDIA A6000 with 48GB VRAM, the VRAM usage of `vit_h` using batch size of 2 requires around 41GB.
- 23-07-20: [**Bug fixed**] `eval_callback` is now available, but is pretty time-consuming (300 mins for ~1600 images, `vit_h`), so we still recommend leaving `eval_flag` to `False` to boost the efficiency of training.
- 23-07-21: `get_miou_png` method in class `eval_callback` has been rewritten according to `Sam.forward()` to speed up inference based on batching input images. It now produces masks 20x faster (14 mins for ~1600 images, `vit_h`) compared to previous versions using  `SamAutomaticMaskGenerator`.
  - The preprocess, post-process and prompt embedding in `Sam.forward()` are still not implmented in this process.
- 23-07-22: Code is slighty modified to train for **crater segmentation**


