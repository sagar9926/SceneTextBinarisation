# SceneTextBinarisation

## Problem Statement : 

Scene text binarization is a difficult task due to text differences in the background noise, and uneven lighting in Natural Scene pictures. In our work we try to improve upon the existing Otsu’s and Kapur Sahoo Wong's binarization method that leverages class variances of image histogram of the image to classify the pixel into background and foreground based. 

## Proposed Solutions : 

### Method 1 : 
We propose a modified Otsu’s method that improves upon the existing method by incorporating Gaussian Blur to reduce noise, enhances the contrast of the image using Histogram Equalization and next apply sharpening. Finally, it applies the Otsu’s binarization using a sliding window technique such that binarization is performed on local context of image instead of the global context. 

### Method 2 : 
Kapur Sahoo Wong method maximizes entropy of image’s histogram to get optimum threshold. Here we applied Gaussian Blur to reduce noise next applied Non-Local Mean Two Dimensional Histogram Equalization and finally applied Kapur Sahoo Wong's algorithm to get optimum threshold.
