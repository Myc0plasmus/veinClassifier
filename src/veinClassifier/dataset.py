import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2

class EyeDataset(Dataset):
    def __init__(self, img_dir, mask_dir, imgFilter=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.imgFiles = sorted(os.listdir(img_dir))
        self.maskFiles = sorted(os.listdir(mask_dir))
        self.imgFilter = imgFilter

    def __len__(self):
        return len(self.imgFiles)

    def __getitem__(self, idx):
        imgFile = self.imgFiles[idx]
        maskFile = self.maskFiles[idx]

        img = cv2.imread(os.path.join(self.img_dir, imgFile))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        segm = cv2.imread(os.path.join(self.mask_dir, maskFile), cv2.IMREAD_GRAYSCALE)
        
        if self.imgFilter is not None:
            img = self.imgFilter(img)
        
        img = np.array(img) / 255.0
        segm = np.array(segm) / 255.0

        img = np.transpose(img, (2, 0, 1))

        segm = (segm > 0.5).astype(np.float32)

        x_img = torch.tensor(img).float()
        y_segm = torch.tensor(segm).unsqueeze(0).float()

        return x_img, y_segm

if __name__ == "__main__":
    dataset = EyeDataset("data/healthy/", "data/healthy_manualsegm")

    img, mask = dataset[0]

    print(mask.shape)
