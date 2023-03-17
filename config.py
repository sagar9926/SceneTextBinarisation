import os

## Project_directory
proj_dir = './scene_text'

## Scene Images Dir
img_dir = os.path.join(proj_dir,'img')

## Binary Images Dir
binary_result_dir = os.path.join(proj_dir,'binary')

## Ground truth .xml file paths 
train_xml_path = os.path.join(proj_dir,'train.xml')
test_xml_path = os.path.join(proj_dir,'test.xml')

## Extracted ground truth paths
class_label_path = os.path.join(proj_dir,"ground_truth.csv")

## OCR results :

otsu_test_extract_results_path = os.path.join(proj_dir,"results/otsu_test_extract_results_path.csv")
slide_otsu_test_extract_results_path = os.path.join(proj_dir,"results/slide_otsu_test_extract_results_path.csv")


# Directory Creation 
if not os.path.exists(proj_dir):
   os.makedirs(proj_dir)
