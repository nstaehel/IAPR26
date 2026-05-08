import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Callable
from datetime import datetime
import cv2
from case_0 import *

def load_image():
    # Define path
    path_he = "images/L1000915.jpg"
    # Check if folder and image exist
    assert os.path.exists(path_he), "Image not found, please check directory structure"

    # Load image
    img_he = np.array(Image.open(path_he))

    cv2.imshow('Case 0', img_he)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()

    return img_he


def determine_case(img):
    M, N, _ = np.shape(img)

    data_red = np.zeros((M, N))
    data_green = np.zeros((M, N))
    data_blue = np.zeros((M, N))

    data_red = img[:,:,0]
    data_green = img[:,:,1]
    data_blue = img[:,:,2]

    mean_red = np.mean(data_red, axis=(0,1))
    mean_green = np.mean(data_green, axis=(0,1))
    mean_blue = np.mean(data_blue, axis=(0,1))

    mean = (mean_blue + mean_green + mean_red)/3

    print(mean)

    if mean > 200:
        case = 0
    else:
        case = 1

    return case

img = load_image()

case = determine_case(img)

print(case)

if case:
    print("rien")

else:
    black_objects = black_threshold(img)
