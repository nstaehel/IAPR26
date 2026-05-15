import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import skimage as ski
import matplotlib.pyplot as plt
from PIL import Image
from image_utils import *

player_name_order = ["p3", "p2", "p4", "p1"]

def determine_case(img, thresh=200):
    return int(np.mean(img) < thresh)

def find_player(im_obj):
    M, N = np.shape(im_obj)
    mean = np.zeros(4)
    for i in range(2):
        for j in range(2):
            cropped_img = im_obj[i*M//2: M//2+i*M//2, j*N//2:N//2+j*N//2]
            mean[2*i + j] = np.mean(cropped_img, axis=(0,1))
    player_index = np.argmax(mean)
    player = player_name_order[player_index]
    return player

def detection_case_0(img, display=False):
    size_dilation = 5
    size_erosion = 50
    r_thresh = (40, 140)
    g_thresh = (40, 140)
    b_thresh = (40, 140)
    
    img_thresh = rgb_threshold(img, r_thresh, g_thresh, b_thresh)
    img_obj_dilation = ski.morphology.isotropic_dilation(img_thresh, size_dilation)
    img_obj = ski.morphology.isotropic_erosion(img_obj_dilation, size_erosion)
    player = find_player(img_obj)
    if (display):
        plt.imshow(img_thresh)
        plt.show()
        plt.imshow(img_obj_dilation)
        plt.show()
        plt.imshow(img_obj)
        plt.show()
    return player

def detection_case_1(img, display=False):
    size_dilation = 5
    size_erosion = 50
    
    h_thresh = (0.1, 0.17)
    s_thresh = (0.4, 1.0)
    v_thresh = (0.9, 1.0)
    img_thresh = hsv_threshold(img, h_thresh, s_thresh, v_thresh)
    
    img_obj_dilation = ski.morphology.isotropic_dilation(img_thresh, size_dilation)
    img_obj = ski.morphology.isotropic_erosion(img_obj_dilation, size_erosion)
    player = find_player(img_obj)
    if (display):
        plt.imshow(img_thresh)
        plt.show()
        plt.imshow(img_obj_dilation)
        plt.show()
        plt.imshow(img_obj)
        plt.show()
    return player

def detection_player(img, display=False):
    case = determine_case(img)
    player = 'p1'
    if (case == 0):
        player = detection_case_0(img, display=display)
    else:
        player = detection_case_1(img, display=display)
    if (display):
        print(case)
        print(player)
    return player

#------------------------------------------------------------------------

def contours_smoothing(img, size_closing=10.0, size_dilation=10.0, size_remove_objects=1500):
    img = ski.morphology.isotropic_closing(img, size_closing)
    img = ski.morphology.remove_small_objects(img, min_size=size_remove_objects)
    img = ski.morphology.isotropic_dilation(img, size_dilation)

    img_binary = (img * 255).astype(np.uint8)
    img_smooth = cv2.medianBlur(img_binary, 5)
    return img_smooth

def card_mask(img):
    img_yellow = hsv_threshold(img, (0.10, 0.17), (0.4, 1.0), (0.9, 1.0))
    img_red_low = hsv_threshold(img, (0.0, 0.08), (0.7, 1.0), (0.7, 1.0))
    img_red_up = hsv_threshold(img, (0.95, 1.0), (0.7, 1.0), (0.85, 1.0))
    img_blue = hsv_threshold(img, (0.5, 0.6), (0.35, 1.0), (0.65, 1.0))
    img_green = hsv_threshold(img, (0.25, 0.45), (0.35, 1.0), (0.55, 1.0))
    img_black = hsv_threshold(img, (0.0, 0.1), (0.0, 0.4), (0.0, 0.5))

    img_card_mask = (img_yellow | img_red_low | img_red_up | img_blue | img_green | img_black)
    img_card_mask_smooth = contours_smoothing(img_card_mask)
    return img_card_mask_smooth
    
def generate_backgound_image(img_dir):
    images_case1 = []
    for image_file in os.listdir(img_dir):
        image_path = os.path.join(img_dir, image_file)
        img = load_image(image_path, display=False)
        if (determine_case(img) == 1):
            images_case1.append(img)
    images_case1 = np.array(images_case1)
    return np.median(images_case1, axis=0).astype(np.uint8)

def fade_background(img, background, threshold=0.015):
    if (determine_case(img) == 0):
        return img
    img_hsv = ski.color.rgb2hsv(img.copy())
    background_hsv = ski.color.rgb2hsv(background.copy())
    absolute_diff_img = np.abs(img_hsv[:,:,0] - background_hsv[:,:,0])
    background_mask = (absolute_diff_img < threshold)
    img_modified = img.copy()
    img_modified[background_mask] = 255
    return img_modified

def remove_background(img, background):
    img = img.copy()
    img_fade_background = fade_background(img, background)
    background_mask = np.logical_not(card_mask(img_fade_background))
    img[background_mask] = 255
    return img