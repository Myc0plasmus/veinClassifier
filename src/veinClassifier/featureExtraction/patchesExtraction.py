import numpy as np
import torch
from tqdm import tqdm
from veinClassifier.dataset import EyeDataset

def extractImagePatches(img, kernel_size=5):
    img = img.unsqueeze(0)  

    patches = torch.nn.functional.unfold(
        img,
        kernel_size=kernel_size,
        padding=kernel_size // 2  
    )


    patches = patches.squeeze(0).transpose(0, 1)

    return patches

def extractTargets(segm):
    return segm.squeeze(0).reshape(-1)


def extractDatasetPatches(dataset, MAX_TOTAL_SAMPLES: int = 5000000):
    sample_patches = []
    sample_targets = []
    total_samples = 0

    for img, segm in tqdm(dataset):
        patches = extractImagePatches(img)
        targets = extractTargets(segm)

        pos_idx = (targets == 1).nonzero(as_tuple=True)[0]
        neg_idx = (targets == 0).nonzero(as_tuple=True)[0]

        if len(pos_idx) == 0:
            continue

        num_pos = len(pos_idx)

        num_neg = min(len(neg_idx), num_pos)
        neg_sample = neg_idx[torch.randperm(len(neg_idx))[:num_neg]]

        idx = torch.cat([pos_idx, neg_sample])
        remaining = MAX_TOTAL_SAMPLES - total_samples
        if MAX_TOTAL_SAMPLES > 0 and remaining <= 0:
            break
        
        patches = patches[idx]
        targets = targets[idx]

        sample_patches.append(patches)
        sample_targets.append(targets)

        total_samples += len(idx)

    sample_patches = torch.cat(sample_patches, dim=0)
    sample_targets = torch.cat(sample_targets, dim=0)

    return sample_patches, sample_targets


