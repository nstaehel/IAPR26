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

    img_thresh_h_up = data_h < 0.45
    img_thresh_s_up = data_s <= 1
    img_thresh_v_up = data_v <= 1

    img_thresh_h_low = data_h > 0.25
    img_thresh_s_low = data_s > 0.35
    img_thresh_v_low = data_v > 0.55

    img_thresh_temp = img_thresh_h_up & img_thresh_s_up & img_thresh_v_up & img_thresh_h_low & img_thresh_s_low & img_thresh_v_low

    img_morpho_closing = ski.morphology.closing(img_thresh_temp, ski.morphology.disk(6.0))

    size = 1000

    img_thresh = ski.morphology.remove_small_objects(img_morpho_closing, min_size=size)

    return img_thresh

def rgb_black_threshold(img):
    M, N, _ = np.shape(img)

    # Define default values for RGB channels
    data_red = np.zeros((M, N))
    data_green = np.zeros((M, N))
    data_blue = np.zeros((M, N))

    data_red = img[:,:,0]
    data_green = img[:,:,1]
    data_blue = img[:,:,2]

    img_thresh_red_up = data_red < 40
    img_thresh_green_up = data_green < 40
    img_tresh_blue_up = data_blue < 40

    img_thresh_red_low = data_red >= 0
    img_thresh_green_low = data_green >= 0
    img_tresh_blue_low = data_blue >= 0

    img_thresh = img_thresh_red_up & img_thresh_green_up & img_tresh_blue_up & img_thresh_red_low & img_thresh_green_low & img_tresh_blue_low

    return img_thresh

def hsv_black_threshold(img):

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

    img_thresh_h_up = data_h <= 0.1
    img_thresh_s_up = data_s <= 0.4
    img_thresh_v_up = data_v <= 0.5

    img_thresh_h_low = data_h >= 0
    img_thresh_s_low = data_s >=0
    img_thresh_v_low = data_v >= 0

    img_thresh_temp = img_thresh_h_up & img_thresh_s_up & img_thresh_v_up & img_thresh_h_low & img_thresh_s_low & img_thresh_v_low

    img_morpho_closing = ski.morphology.closing(img_thresh_temp, ski.morphology.disk(2.0))

    size = 2500

    img_thresh = ski.morphology.remove_small_objects(img_morpho_closing, min_size=size)

    return img_thresh

def contours_smoothing(img):

    img_binary = (img * 255).astype(np.uint8)


    img_smooth = cv2.medianBlur(img_binary, 5)

    return img_smooth

def preprocessing(img):
    "Threshold sur rouge, vert, bleu, jaune et noir et clean images si besoin du style median filter et morphology"

    img_yellow = hsv_yellow_threshold(img)
    img_red = hsv_red_threshold(img)
    img_blue = hsv_blue_threshold(img)
    img_green = hsv_green_threshold(img)
    img_black = hsv_black_threshold(img)

    img_yellow_smooth = contours_smoothing(img_yellow)
    img_red_smooth = contours_smoothing(img_red)
    img_blue_smooth = contours_smoothing(img_blue)
    img_green_smooth = contours_smoothing(img_green)
    img_black_smooth = contours_smoothing(img_black)


    return img_yellow_smooth, img_red_smooth, img_blue_smooth, img_green_smooth, img_black_smooth