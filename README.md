# SceneTextBinarisation

## Problem Statement : 

Scene text binarization is a difficult task due to text differences in the background noise, and uneven lighting in Natural Scene pictures. In our work we try to improve upon the existing Otsu’s and Kapur Sahoo Wong's binarization method that leverages class variances of image histogram of the image to classify the pixel into background and foreground based. 

## Proposed Solutions : 

### Method 1 : 
We propose a modified Otsu’s method that improves upon the existing method by incorporating Gaussian Blur to reduce noise, enhances the contrast of the image using Histogram Equalization and next apply sharpening. Finally, it applies the Otsu’s binarization using a sliding window technique such that binarization is performed on local context of image instead of the global context. 

### Method 2 : 
Kapur Sahoo Wong method maximizes entropy of image’s histogram to get optimum threshold. Here we applied Gaussian Blur to reduce noise next applied Non-Local Mean Two Dimensional Histogram Equalization and finally applied Kapur Sahoo Wong's algorithm to get optimum threshold.


## Performance Evaluation :

* Run the algorithm to convert the Scene Text images to Binary images 
* Use off-the shelf OCR technique, for our project we used keras-ocr to extract text from binary images
* Calculate similarity score ratio measure the similarity of extracted text with the ground truth and define a similarity score

For evaluating the quality of extraction we have used fuzzy logic for string matching. We have calculated the __partial ratio__ raw score which is a measure of the strings similarity.The partial ratio helps us to perform substring matching. This takes the shortest string and compares it with all the substrings of the same length. This helps us to identify if the extracted result is closer to the ground truth

Once the text is extracted from the binary images using **Keras-OCR** method. Then for evaluating the quality of extraction we have used fuzzy logic for string matching.We have calculated the partial ratio raw score which is a measure of the strings similarity and used the same to compare the performance of the Slide Otsu Technique with the classical otsu.

For our project we have tested our algorithms on 322+ images from SVT dataset. It is a dataset that was harvested from Google Street View. Following table compares the similarity score distribution of Otsu vs Slide Otsu. In our results we can see that with the pro-
posed Slide Otsu method, the average similarity with the ground truth has improved by 5 % from 57.27 % to 62.1 % that too with a lesser standard deviation as compared to the Original Otsu.

![alt text](https://github.com/sagar9926/SceneTextBinarisation/blob/main/results/Slide_Otsu/sotsu_res3.png)

## Reults :

### Slide Otsu : 

#### Example 1 :

![alt text](https://github.com/sagar9926/SceneTextBinarisation/blob/main/results/Slide_Otsu/sotsu_res1.png)

#### Example 2 :

![alt text](https://github.com/sagar9926/SceneTextBinarisation/blob/main/results/Slide_Otsu/sotsu_res2.png)

#### Example 3 : 

![alt text](https://github.com/sagar9926/SceneTextBinarisation/blob/main/results/Slide_Otsu/sotsu_res3.png)
