import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2

class EyeDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.imgFiles = sorted(os.listdir(img_dir))
        self.maskFiles = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.imgFiles)

    def __getitem__(self, idx):
        imgFile = self.imgFiles[idx]
        maskFile = self.maskFiles[idx]

        img = cv2.imread(os.path.join(self.img_dir, imgFile))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, maskFile), cv2.IMREAD_GRAYSCALE)

        
        img = np.array(img) / 255.0
        mask = np.array(mask) / 255.0

        img = np.transpose(img, (2, 0, 1))

        mask = (mask > 0.5).astype(np.float32)

        img = torch.tensor(img).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return img, mask
        # return img.float(), mask.unsqueeze(0).float()
