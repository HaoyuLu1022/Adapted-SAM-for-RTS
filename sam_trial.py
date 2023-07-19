from segment_anything import sam_model_registry
device = "cuda:1"
sam = sam_model_registry["vit_h"](checkpoint="model_data/sam_vit_h_4b8939.pth").to(device)
print(sam)