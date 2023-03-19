
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

## Input Paths
from config import class_label_path

## Output Paths 
from config import img_dir , binary_result_dir ,otsu_text_extract_results_path,slide_otsu_text_extract_results_path

## Helper Function
from evaluation_fn import evaluation_metric

class SlideOtsu :
  def __init__(self,image_path,binary_result_dir,method) :
    
    """
    This is a class that implements the proposed SlideOtsu method.
    """
    # Read RGB image
    self.image_path = image_path
    if method == 'slide_otsu':
      self.binary_result_dir = os.path.join(binary_result_dir,'slide_otsu')
    elif method == 'otsu' :
       self.binary_result_dir = os.path.join(binary_result_dir,'otsu')
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

  ## Otsu Run
  print("Running Otsu's Binarisation : ")
  for image_path in os.listdir(img_dir) :
    slide_otsu = SlideOtsu(os.path.join(img_dir,image_path),binary_result_dir,method = 'otsu')
    binary_img= slide_otsu.run_otsu()
  result = slide_otsu.ocr_extract()
  
  df_otsu = pd.DataFrame(result,columns = ['image_path','extracted_text'])
  df_otsu['image_path'] = df_otsu['image_path'].apply(lambda text : text.replace('./scene_text/binary/otsu/',''))
  df_otsu['image_path'] = df_otsu['image_path'].apply(lambda text : text.replace('binary_',''))
  df_otsu['extracted_text'] = df_otsu['extracted_text'].apply(lambda text : text.upper())
  
  ## Storing Otsu Results
  df_otsu.to_csv(otsu_text_extract_results_path,index = False)  
  

  ## SlideOtsu Run
  print("Running Slide_Otsu's Binarisation : ")
  for image_path in os.listdir(img_dir) :
    slide_otsu = SlideOtsu(os.path.join(img_dir,image_path),binary_result_dir,method = 'slide_otsu')
    binary_img= slide_otsu.run_slide_otsu()
  result = slide_otsu.ocr_extract()
  
  df_slide_otsu = pd.DataFrame(result,columns = ['image_path','extracted_text'])
  df_slide_otsu['image_path'] = df_slide_otsu['image_path'].apply(lambda text : text.replace('./scene_text/binary/slide_otsu/',''))
  df_slide_otsu['image_path'] = df_slide_otsu['image_path'].apply(lambda text : text.replace('binary_',''))
  df_slide_otsu['extracted_text'] = df_slide_otsu['extracted_text'].apply(lambda text : text.upper())
  
  ## Storing SlideOtsu Results
  df_slide_otsu.to_csv(slide_otsu_text_extract_results_path,index = False)
  
  ## Performance Evaluation
  
  ### Read the ground truth file
  df_gt = pd.read_csv(class_label_path)
  
  print("Evaluating Model Performances ... ")
  df_merge = pd.merge(df_otsu,df_slide_otsu,on = 'image_path',how = 'inner',suffixes = ("_otsu",'_slide_otsu'))
  df_merge = pd.merge(df_merge,df_gt,on = 'image_path')
  
  df_merge =  evaluation_metric(df_merge,ground_truth_col = 'scene_text' , pred_col = 'extracted_text_otsu',method = 'otsu') 
  df_merge =  evaluation_metric(df_merge,ground_truth_col = 'scene_text' , pred_col = 'extracted_text_slide_otsu',method = 'slide_otsu')
  
  print("Comparison of text similarity scores for the two methods (Otsu v/s Slide_Otsu): ")
  print(df_merge[['score_otsu','score_slide_otsu']].describe())
  
  



