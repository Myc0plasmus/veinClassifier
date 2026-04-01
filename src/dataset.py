import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class EyeDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.imgFiles = os.listdir(img_dir)
        self.maskFiles = os.listdir(mask_dir)

    def __len__(self):
        return len(self.imgFiles)

    def __getitem__(self, idx):
        imgFile = self.imgFiles[idx]
        maskFile = self.maskFiles[idx]

        img = Image.open(os.path.join(self.img_dir, imgFile)).convert("L")
        mask = Image.open(os.path.join(self.mask_dir, maskFile)).convert("L")

        # img = np.array(img) / 255.0
        # mask = np.array(mask) / 255.0

        # binary mask (important!)
        # mask = (mask > 0.5).astype(np.float32)

        # img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        # mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return img, mask

