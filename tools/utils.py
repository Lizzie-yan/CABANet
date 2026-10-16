import numpy as np


def label_to_RGB(image):
    RGB = np.zeros(shape=[image.shape[0], image.shape[1], 3], dtype=np.uint8)
    index = image == 0
    RGB[index] = np.array([162, 202, 253])
    index = image == 1
    RGB[index] = np.array([222, 222, 222])
    return RGB