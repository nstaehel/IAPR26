import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import v2

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