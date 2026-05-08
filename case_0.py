import numpy as np
import skimage as ski


def color_threshold(img):
    M, N, _ = np.shape(img)

    # Define default values for RGB channels
    data_red = np.zeros((M, N))
    data_green = np.zeros((M, N))
    data_blue = np.zeros((M, N))

    data_red = img[:,:,0]
    data_green = img[:,:,1]
    data_blue = img[:,:,2]

    img_thresh_red_up = data_red < 140
    img_thresh_green_up = data_green < 140
    img_tresh_blue_up = data_blue < 140

    img_thresh_red_low = data_red > 40
    img_thresh_green_low = data_green > 40
    img_tresh_blue_low = data_blue > 40

    img_thresh = img_thresh_red_up & img_thresh_green_up & img_tresh_blue_up & img_thresh_red_low & img_thresh_green_low & img_tresh_blue_low

    return img_thresh

def size_threshold(img, size):
    img_obj = np.zeros_like(img)

    img_obj = ski.morphology.remove_small_objects(img, min_size = size)

    return img_obj

def find_player(img):

    M, N = np.shape(img)

    mean = np.zeros(4)
    a = 0

    for i in range(2):
        for j in range(2):

            cropped_img = img[i*M//2: M//2+i*M//2, j*N//2:N//2+j*N//2]

            #mean[0]=p3, mean[1]=p2, mean[2]= p4, mean[3]=p1

            mean[a]=np.mean(cropped_img, axis=(0,1))
            a += 1

    player = np.argmax(mean)

    return player

def player_name(player):

    if(player == 0):
        return "p3"
    if(player == 1):
        return "p2"
    if(player == 2):
        return "p4"
    if(player == 3):
        return "p1"

def detection_case_0(img):

    img_thresh = color_threshold(img)

    size = 30000

    img_obj = size_threshold(img_thresh, size)

    player = find_player(img_obj)

    player_nbr = player_name(player)

    return player_nbr