from skimage import exposure
from skimage.filters import gaussian
from skimage.filters import frangi
import numpy as np

import cv2
from skimage.morphology import remove_small_objects, remove_small_holes

def complexInputFilter(image):
    # green chanel
    green = image[:, :, 1]
    green = 255 - green

    # CLAHE
    enhanced = exposure.equalize_adapthist(green)

    # Gaussian smoothing
    smooth = gaussian(enhanced, sigma=1)

    # FRANGI
    vessel = frangi(smooth, sigmas=range(1, 5), black_ridges=True)

    vessel = (vessel - vessel.min()) / (vessel.max() - vessel.min() + 1e-8)

    # overlay instead of replace
    rgb = image.astype(np.float32) / 255.0

    rgb_overlay = rgb.copy()
    rgb_overlay[:,:,1] = np.maximum(rgb[:,:,1], vessel)  # enhance green channel
    rgb_overlay[:,:,0] = rgb[:,:,0] * 0.3 
    rgb_overlay[:,:,2] = rgb[:,:,2] * 0.7 

    return rgb_overlay

def complexOutputFilter(segm, size=50):
    mask_bool = segm.astype(bool)

    cleaned = remove_small_objects(mask_bool, max_size = size)

    cleaned = remove_small_holes(cleaned, max_size = size)

    cleaned = cleaned.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return cleaned
