import os
import random
from PIL import Image
import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import v2
import cv2

def get_random_imagepath(img_dir, annotations):
    image_id_list = list(annotations['image_id'])
    image_id = random.choice(image_id_list)
    return os.path.join(img_dir, image_id) + '.jpg'

def load_image(image_path, display=True):
    assert os.path.exists(image_path), "Image not found, please check directory structure"
    img = np.array(Image.open(image_path))
    if (display):
        print(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    return img

def rgb_threshold(img, r_thresh, g_thresh, b_thresh):
    img = img.copy()
    img_r = img[:,:,0]
    img_g = img[:,:,1]
    img_b = img[:,:,2]
    img_thresh_r = (img_r >= r_thresh[0]) & (img_r <= r_thresh[1])
    img_thresh_g = (img_g >= g_thresh[0]) & (img_g <= g_thresh[1])
    img_thresh_b = (img_b >= b_thresh[0]) & (img_b <= b_thresh[1])
    img_thresh = img_thresh_r & img_thresh_g & img_thresh_b
    return img_thresh

def hsv_threshold(img, h_thresh, s_thresh, v_thresh):
    img = ski.color.rgb2hsv(img.copy())
    img_h = img[:,:,0]
    img_s = img[:,:,1]
    img_v = img[:,:,2]
    img_thresh_h = (img_h >= h_thresh[0]) & (img_h <= h_thresh[1])
    img_thresh_s = (img_s >= s_thresh[0]) & (img_s <= s_thresh[1])
    img_thresh_v = (img_v >= v_thresh[0]) & (img_v <= v_thresh[1])
    img_thresh = img_thresh_h & img_thresh_s & img_thresh_v
    return img_thresh

def contours_smoothing(img, size_closing=6.0, size_dilation=10.0, size_remove_objects=1500):
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

def divide_image(image, index):
    # Crop the image into 5 cropped images of the same size (given as tensor): 
    # 'center_card', 'player_1_cards', 'player_2_cards', 'player_3_cards', 'player_4_cards'
    colors, height, width = image.shape
    height_half_d = height // 6
    width_half_d = (width - 4*height_half_d) // 2
    centers = [[height // 2, width // 2], 
               [height - height_half_d, width // 2], 
               [height // 2, width - height_half_d], 
               [height_half_d, width // 2], 
               [height // 2, height_half_d]]
    if (index == 2) or (index == 4):
        height_half_d = width_half_d
        width_half_d = height // 6
    cropped_image = image[:, centers[index][0]-height_half_d:centers[index][0]+height_half_d, centers[index][1]-width_half_d:centers[index][1]+width_half_d]
    cropped_image = cropped_image.clone()
    if (index == 2) or (index == 4):
        cropped_image = cropped_image.transpose(-1, -2)
    return cropped_image

def display_tensor_images(imgs, transform=None, shape=None, figsize=(5,5)):
    # Shape is a tuple of (nb_image_horizontal, nb_image_vertical) such that len(imgs) = nb_image_vertical*nb_image_horizontal
    # Otherwise the images are displayed in a list (horizontally)
    if (not isinstance(imgs, list)):
        imgs = [imgs]
    if (shape == None):
        shape = (1, len(imgs))
    shape_x, shape_y = shape
    fig, ax = plt.subplots(nrows=shape_y, ncols=shape_x, squeeze=False)
    fig.set_size_inches(figsize[0], figsize[1])
    for i, img in enumerate(imgs):
        img = img.clone()
        if (transform != None):
            img = transform(img)
        img = v2.functional.to_pil_image(img)
        img_x = i % shape_x
        img_y = i // shape_x
        ax[img_y, img_x].imshow(np.asarray(img))
        ax[img_y, img_x].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

def display_rgb_hist(img):
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    img_red = img.copy()[:,:,0]
    img_green = img.copy()[:,:,1]
    img_blue = img.copy()[:,:,2]
    # Display histograms
    plt.hist(img_red.ravel(), bins=256)
    plt.xlabel('Pixel intensity')
    plt.ylabel('Number of pixels')
    plt.title('Red')
    plt.xticks(np.arange(0, 256, 10), rotation=45)
    plt.show()
    print(img_red.mean())
    
    plt.hist(img_green.ravel(), bins=256)
    plt.xlabel('Pixel intensity')
    plt.ylabel('Number of pixels')
    plt.title('Green')
    plt.xticks(np.arange(0, 256, 10), rotation=45)
    plt.show()
    print(img_green.mean())
    
    plt.hist(img_blue.ravel(), bins=256)
    plt.xlabel('Pixel intensity')
    plt.ylabel('Number of pixels')
    plt.title('Blue')
    plt.xticks(np.arange(0, 256, 10), rotation=45)
    plt.show()
    print(img_blue.mean())