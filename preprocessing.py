import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def resize_image(img, size):
    return cv2.resize(img, size)


# In[4]:


def load_img(image_path):
    images = []
    for img in os.listdir(image_path):
        image = cv2.imread(os.path.join(image_path, img))
        image = cv2.resize(image, (256, 256))
        images.append(image)
    return np.array(images)


def plot_img(image):
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()


def convert_to_gray(images):
    gray_img = []
    for image in images:
        gray_img.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return np.array(gray_img)


def get_gaussian_blur(images):
    gaus = []
    for img in images:
        gaus.append(cv2.GaussianBlur(img, (11, 11), 11))
    return np.array(gaus)


def equalize_hist(images):
    hist = []
    for img in images:
        hist.append(cv2.equalizeHist(img))
    return np.array(hist)
