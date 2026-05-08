import numpy as np
import skimage as ski
import cv2


def hsv_threshold(img):

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

def size_threshold(img, size):
    img_obj = np.zeros_like(img)

    img_obj = ski.morphology.remove_small_objects(img, min_size = size)

    binary = (img_obj * 255).astype(np.uint8)

    contour, _ = cv2.findContours(binary, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    return contour


def position_contours(contours):

    N = len(contours)

    position = np.zeros((N, 2))

    for i, cnt in enumerate(contours):
        position[i], _ = cv2.minEnclosingCircle(cnt)
        
    return position

def verification(position, pos_player):

    wrong_player = False

    for pos in position:
        if (np.abs(pos_player[0] - pos[0]) <= 50) or (np.abs(pos_player[1] - pos[1]) <= 50):

            if (pos_player[0] == pos[0]) and (pos_player[1] == pos[1]):
                wrong_player = False
            else:
                wrong_player = True
    
    return wrong_player

def wrong_circle(scores, position, pos_player, idx):

    scores_sorted = np.sort(scores)

    if ((scores_sorted[1] - scores_sorted[0]) <= 0.05):

        wrong_object = verification(position, pos_player)   

        if wrong_object:
            wrong_player = scores[idx]

            scores.remove(wrong_player)

            idx_new = np.argmin(scores)

            if idx_new >= idx:
                #to keep the correct order for the position
                idx_new += 1
            
            pos_player = position[idx_new, :]
    
    return pos_player

def comparison_circle(contours):

    position = position_contours(contours)
    pos_player = np.zeros(2)
    inertia_weight = 0.7
    circ_weight = 0.3

    circularity = 0
    scores = []


    for cnt in (contours): 

        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)

        if perimeter == 0 or M["m00"] == 0:
            scores.append(float('inf'))
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        circ_score = (1 - circularity) ** 2  # 0 = parfait
        
        
        mu20 = M["mu20"] / M["m00"]
        mu02 = M["mu02"] / M["m00"]
        mu11 = M["mu11"] / M["m00"]
        inertia_ratio = (mu20 - mu02)**2 + 4 * mu11**2  #if circle, mu20 = mu02 and mu11=0 bc no coupling between axes
        inertia_sum = (mu20 + mu02)**2                  #normalize to be invariant in scaling

        if inertia_sum > 0:
            inertia_score = inertia_ratio / inertia_sum
        else:
            inertia_score = float('inf')


        #to have a perfect disk : circ = 0 and inertia = 0
        total = circ_weight*circ_score + inertia_weight*inertia_score

        scores.append(total)
    
    idx = np.argmin(scores)
    
    pos_player = position[idx, :]


    if len(scores) > 1:

        pos_player = wrong_circle(scores, position, pos_player, idx)

    return pos_player

def player_number(pos_player, M, N):

    if pos_player[0] <= N/2 and pos_player[1] <= M/2:
        player_nbr = "p3"
    if pos_player[0] <= N/2 and pos_player[1] > M/2:
        player_nbr = "p4"
    if pos_player[0] > N/2 and pos_player[1] <= M/2:
        player_nbr = "p2"
    if pos_player[0] > N/2 and pos_player[1] > M/2:
        player_nbr = "p1"
    
    return player_nbr

def detection_case_1(img, M, N):
    img_thresh = hsv_threshold(img)

    size = 10000

    obj = size_threshold(img_thresh, size=size)

    player_pos = comparison_circle(obj)

    player_nbr = player_number(player_pos, M, N)

    return player_nbr