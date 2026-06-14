import numpy as np
import torch

def reshape_patches(patches, C=3, K=5):
    return patches.view(-1, C, K, K) 

def compute_statistics(patches):
    # patches: (N, C, 5, 5)

    mean = patches.mean(dim=(2, 3))           # (N, C)
    std = patches.std(dim=(2, 3))             # (N, C)
    var = patches.var(dim=(2, 3))             # (N, C)

    min_val = patches.amin(dim=(2, 3))
    max_val = patches.amax(dim=(2, 3))

    return torch.cat([mean, std, var, min_val, max_val], dim=1)

def compute_moments(patches):
    mean = patches.mean(dim=(2,3), keepdim=True)
    centered = patches - mean

    m2 = (centered**2).mean(dim=(2,3))   # variance-like
    m3 = (centered**3).mean(dim=(2,3))   # skewness-like
    m4 = (centered**4).mean(dim=(2,3))   # kurtosis-like

    return torch.cat([m2, m3, m4], dim=1)

def rgb_to_gray(patches):
    # simple luminance approximation
    r, g, b = patches[:,0], patches[:,1], patches[:,2]
    gray = 0.299*r + 0.587*g + 0.114*b
    return gray

import cv2

def compute_hu_moments(patches):
    gray = rgb_to_gray(patches)
    patches_np = gray.numpy()
    features = []

    for p in patches_np:
        # p: (5,5)
        m = cv2.moments(p.astype(np.float32))
        hu = cv2.HuMoments(m).flatten()
        features.append(hu)

    return torch.tensor(features, dtype=torch.float32)

def compute_statistics_features(patches):
    patches = reshape_patches(patches)

    return compute_moments(patches)

def compute_moment_features(patches):
    patches = reshape_patches(patches)

    return compute_moments(patches)

def compute_hu_moments_features(patches):
    patches = reshape_patches(patches)

    return compute_hu_moments(patches)

def compute_all_features(patches):
    patches = reshape_patches(patches)

    basic = compute_statistics(patches)
    moments = compute_moments(patches)
    hu = compute_hu_moments(patches)

    return torch.cat([basic, moments, hu], dim=1)
