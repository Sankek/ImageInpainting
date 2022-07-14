import os
import numpy as np

import cv2 as cv


def get_files_recursive(path):
    all_files = []
    entered = False
    for root, _, files in os.walk(path):
        entered = True
        for file in files:
            all_files.append(os.path.join(root, file))
    if not entered:
        for file in os.listdir(path):
            all_files.append(os.path.join(path, file))
            
    return all_files


def central_crop_resize(img, size):
    """Crops numpy image to square with size (size, size)"""
    shape = img.shape
    if len(shape) != 3:
        raise NotImplementedError
    img_h, img_w = shape[:2]
    if img_w <= img_h:
        left, upper, right, lower = 0, (img_h-img_w)//2, img_w, (img_h+img_w)//2
    else:
        left, upper, right, lower = (img_w-img_h)//2, 0, (img_w+img_h)//2, img_h

    img = img[upper:lower, left:right, :]
    
    interpolation = cv.INTER_CUBIC if img_h <= size else cv.INTER_AREA
    img = cv.resize(img, (size, size), interpolation=interpolation)

    return img


def preprocess_data(walk_path, save_path, image_size):
    """Gather all images from directory recursively and resize them to (image_size, image_size)"""
    images = get_files_recursive(walk_path)
    np.random.shuffle(images)

    for i, image_path in enumerate(images):
        try:
            img = cv.imread(image_path)
        except cv.error as e:
            print(image_path)
            print(e)
            continue
        
        img = central_crop_resize(img, image_size)
        
        save_name = f'{i}.jpg'
        cv.imwrite(os.path.join(save_path, save_name), img)
