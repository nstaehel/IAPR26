import numpy as np
import skimage as ski
import cv2

def hsv_yellow_threshold(img):

    M, N, C = np.shape(img)

    data_h = np.zeros((M, N))
    data_s = np.zeros((M, N))
    data_v = np.zeros((M, N))

    # ------------------
    # Your code here ... 
    # ------------------
    img_hsv = ski.color.rgb2hsv(img)
    data_h = img_hsv[:,:,0]
    data_s = img_hsv[:,:,1]
    data_v = img_hsv[:,:,2]

    img_thresh_h_up = data_h < 0.17
    img_thresh_s_up = data_s <= 1
    img_thresh_v_up = data_v <= 1

    img_thresh_h_low = data_h > 0.10
    img_thresh_s_low = data_s > 0.4
    img_thresh_v_low = data_v > 0.9

    img_thresh = img_thresh_h_up & img_thresh_s_up & img_thresh_v_up & img_thresh_h_low & img_thresh_s_low & img_thresh_v_low

    return img_thresh

def hsv_red_threshold(img):

    M, N, C = np.shape(img)

    data_h = np.zeros((M, N))
    data_s = np.zeros((M, N))
    data_v = np.zeros((M, N))

    # ------------------
    # Your code here ... 
    # ------------------
    img_hsv = ski.color.rgb2hsv(img)
    data_h = img_hsv[:,:,0]
    data_s = img_hsv[:,:,1]
    data_v = img_hsv[:,:,2]

    #img_thresh_h_up = data_h > 0.94

    img_thresh_h_low = (data_h >= 0.0) & (data_h < 0.08)
    img_thresh_s_low = (data_s > 0.65) & (data_s <= 1)
    img_thresh_v_low = (data_v > 0.7) & (data_v <= 1)

    #img_thresh_h = img_thresh_h_up | img_thresh_h_low

    img_thresh_h_up = (data_h > 0.95) & (data_h <= 1.0)
    img_thresh_s_up = (data_s > 0.70) & (data_s <= 1)
    img_thresh_v_up = (data_v > 0.85) & (data_v <=1)

    img_thresh_up = img_thresh_h_up & img_thresh_s_up & img_thresh_v_up

    img_thresh_low = img_thresh_h_low & img_thresh_s_low & img_thresh_v_low

    img_thresh_temp = img_thresh_up | img_thresh_low

    size = 1500

    img_morpho_closing = ski.morphology.closing(img_thresh_temp, ski.morphology.disk(6.5))

    img_morpho_so = ski.morphology.remove_small_objects(img_morpho_closing, min_size=size)


    img_thresh = (img_morpho_so * 255).astype(np.uint8)

    return img_thresh

def hsv_blue_threshold(img):

    M, N, C = np.shape(img)

    data_h = np.zeros((M, N))
    data_s = np.zeros((M, N))
    data_v = np.zeros((M, N))

    # ------------------
    # Your code here ... 
    # ------------------
    img_hsv = ski.color.rgb2hsv(img)
    data_h = img_hsv[:,:,0]
    data_s = img_hsv[:,:,1]
    data_v = img_hsv[:,:,2]

    img_thresh_h_up = data_h < 0.60
    img_thresh_s_up = data_s <= 1
    img_thresh_v_up = data_v <= 1

    img_thresh_h_low = data_h > 0.5
    img_thresh_s_low = data_s > 0.35
    img_thresh_v_low = data_v > 0.65

    img_thresh_temp = img_thresh_h_up & img_thresh_s_up & img_thresh_v_up & img_thresh_h_low & img_thresh_s_low & img_thresh_v_low

    img_morpho_closing = ski.morphology.closing(img_thresh_temp, ski.morphology.disk(6.0))

    size = 1500

    img_thresh = ski.morphology.remove_small_objects(img_morpho_closing, min_size=size)

    return img_thresh

def hsv_green_threshold(img):

    M, N, C = np.shape(img)

    data_h = np.zeros((M, N))
    data_s = np.zeros((M, N))
    data_v = np.zeros((M, N))

    # ------------------
    # Your code here ... 
    # ------------------
    img_hsv = ski.color.rgb2hsv(img)
    data_h = img_hsv[:,:,0]
    data_s = img_hsv[:,:,1]
    data_v = img_hsv[:,:,2]

    img_thresh_h_up = data_h < 0.60
    img_thresh_s_up = data_s <= 1
    img_thresh_v_up = data_v <= 1

    img_thresh_h_low = data_h > 0.5
    img_thresh_s_low = data_s > 0.35
    img_thresh_v_low = data_v > 0.65

    img_thresh_temp = img_thresh_h_up & img_thresh_s_up & img_thresh_v_up & img_thresh_h_low & img_thresh_s_low & img_thresh_v_low

    img_morpho_closing = ski.morphology.closing(img_thresh_temp, ski.morphology.disk(6.0))

    size = 1500

    img_thresh = ski.morphology.remove_small_objects(img_morpho_closing, min_size=size)

    return img_thresh


def preprocessing(img):
    "Threshold sur rouge, vert, bleu, jaune et noir et clean images si besoin du style median filter et morphology"

    img_yellow = hsv_yellow_threshold(img)
    img_red = hsv_red_threshold(img)
    img_blue = hsv_blue_threshold(img)


    return img_yellow, img_red, img_blue