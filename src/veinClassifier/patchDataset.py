import numpy as np
from torch.utils.data import Dataset

class PatchDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        patch_size=256,
        vessel_prob=0.8
    ):
        self.base_dataset = base_dataset
        self.patch_size = patch_size
        self.vessel_prob = vessel_prob

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, mask = self.base_dataset[idx]

        p = self.patch_size

        H = img.shape[1]
        W = img.shape[2]

        if (
            np.random.rand() < self.vessel_prob
            and mask.sum() > 0
        ):
            vessel_pixels = np.argwhere(mask[0].numpy() > 0)

            cy, cx = vessel_pixels[
                np.random.randint(len(vessel_pixels))
            ]

            y = cy - p // 2
            x = cx - p // 2

            y = np.clip(y, 0, H - p)
            x = np.clip(x, 0, W - p)

        else:
            y = np.random.randint(0, H - p + 1)
            x = np.random.randint(0, W - p + 1)

        img_patch = img[:, y:y+p, x:x+p]
        mask_patch = mask[:, y:y+p, x:x+p]

        return img_patch, mask_patch
