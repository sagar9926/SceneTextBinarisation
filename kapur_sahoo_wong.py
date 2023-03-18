import cv2
import os
import numpy as np


def get_kapur_threshold(image):
    hist, _ = np.histogram(image, bins=range(256), density=True)
    c_hist = hist.cumsum()
    c_hist_i = 1.0 - c_hist
    c_hist[c_hist <= 0] = 1
    c_hist_i[c_hist_i <= 0] = 1
    c_entropy = (hist * np.log(hist + (hist <= 0))).cumsum()
    b_entropy = -c_entropy / c_hist + np.log(c_hist)
    c_entropy_i = c_entropy[-1] - c_entropy
    f_entropy = -c_entropy_i / c_hist_i + np.log(c_hist_i)
    return np.argmax(b_entropy + f_entropy)


def save_images(images, image_names, dir_name):
    root = os.getcwd()
    path = os.path.join(root, dir_name)
    if not os.path.exists(path):
        os.mkdir(path)
    for img, image in zip(images, image_names):
        cv2.imwrite(os.path.join(path, image), img)
