# importing OpenCV(cv2) module
import cv2
import cv2 as cv
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow  
import numpy as np
import os
import keras_ocr
import matplotlib.pyplot as plt
import pandas as pd
import gc


class SlideOtsu :
  def __init__(self,image_path,method) :
    
    """
    This is a class that implements the proposed SlideOtsu method.
    """
    # Read RGB image
    self.image_path = image_path
    if method == 'slide_otsu':
      self.binary_result_dir = './binary/slide_otsu'
    elif method == 'otsu' :
       self.binary_result_dir = './binary/otsu'
    if not os.path.exists(self.binary_result_dir):
      os.makedirs(self.binary_result_dir)
    self.img = cv2.imread(self.image_path)
    

  def fn_image_enhancement(self) :

    """
    This method implements following chain of image enhancement techniques :
    1. First Grayscale the image
    2. Apply Gaussinan Blur to eliminate the noise
    3. Histogram equalization to enhance the contrast
    4. Sharpening 
    """
    gray_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    ## Gaussian Blur 
    src = cv.GaussianBlur(gray_image, (3, 3), 0)

    ## Histogram Equalisation
    clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(5,5))
    cl1 = clahe.apply(src)

    ## Sharpening 
    # Create the sharpening kernel
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

    # Apply the sharpening kernel to the image using filter2D
    sharpened = cv2.filter2D(cl1, -1, kernel)

    return sharpened

  def sliding_window(self,img, stepSize, windowSize):
    ## Create a copy of the image
    image = img.copy()

    ## Create small square patches and then run Otsu on the local regions
    for y in range(0, image.shape[0], stepSize):
      for x in range(0, image.shape[1], stepSize):
        window = image[y:y + windowSize[1], x:x + windowSize[0]]
        bin_window =  cv.threshold(window,0,255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
        image[y:y + windowSize[1], x:x + windowSize[0]] = bin_window
    return image ,bin_window


  def run_slide_otsu(self) :
    """
    This method implements Executes the Slide Otsu thresholding technique
    
    """
    ## Image Enhancement
    sharpened =  self.fn_image_enhancement()

    # features = []
    win_size = sharpened.shape[0]//35
    
    windows,bin_window = self.sliding_window(sharpened, win_size, (win_size, win_size))

    binary_img_name = self.image_path.split('/')[-1]
    
    cv2.imwrite(os.path.join(self.binary_result_dir,'binary_'+binary_img_name), 255 - windows)
    # cv2.imwrite(os.path.join(self.binary_result_dir,'binary_0_'+binary_img_name), windows)
  
  def run_otsu(self) :

    binary_img_name = self.image_path.split('/')[-1]

    gray_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
    bin_window =  cv.threshold(gray_image,0,255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    
    cv2.imwrite(os.path.join(self.binary_result_dir,'binary_'+binary_img_name), bin_window)

  def ocr_extract(self) :

    image_list = []
    prediction_list = []

    binary_image_path_list = [os.path.join(self.binary_result_dir,b_path) for b_path in os.listdir(self.binary_result_dir)]
    pipeline = keras_ocr.pipeline.Pipeline()

    # Read images from folder path to image object
    for temp_img in binary_image_path_list : 
      images = [
          keras_ocr.tools.read(img) for img in [temp_img]
      ]

      # generate text predictions from the images
      prediction_groups = pipeline.recognize(images)

    # assert len(prediction_groups) == len(binary_image_path_list)

      # for image_name , extracted_result in zip(binary_image_path_list,prediction_groups) :
      
      pred = " ".join([text[0] for text in prediction_groups[0]])
      if len(pred) != 0 :
        prediction_list.append([temp_img,pred])
      gc.collect()
    return(prediction_list)


if __name__ == "__main__" :
  
  img_dir = './img'

  ## Otsu Run
  for image_path in os.listdir(img_dir) :
    slide_otsu = SlideOtsu(os.path.join(img_dir,image_path),method = 'otsu')
    binary_img= slide_otsu.run_otsu()
  result = slide_otsu.ocr_extract()
  
  df_otsu = pd.DataFrame(result,columns = ['image_path','extracted_text'])

  ## SlideOtsu Run
  for image_path in os.listdir(img_dir) :
    slide_otsu = SlideOtsu(os.path.join(img_dir,image_path),method = 'slide_otsu')
    binary_img= slide_otsu.run_slide_otsu()
  result = slide_otsu.ocr_extract()
  
  df_slide_otsu = pd.DataFrame(result,columns = ['image_path','extracted_text'])  
