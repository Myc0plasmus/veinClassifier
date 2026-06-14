from skimage import exposure
from skimage.filters import gaussian
from skimage.filters import frangi

def complexFilter(image):
    # green chanel
    green = image[:, :, 1]

    # CLAHE
    enhanced = exposure.equalize_adapthist(green)

    # Gaussian smoothing
    smooth = gaussian(enhanced, sigma=1)

    # FRANGI
    frangi_img = frangi(smooth)

    return frangi_img
