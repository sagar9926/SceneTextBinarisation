import os

## Project_directory
proj_dir = './scene_text'

## Ground truth .xml file paths 
train_xml_path = os.path.join(proj_dir,'train.xml')
test_xml_path = os.path.join(proj_dir,'test.xml')

## Extracted ground truth paths
class_label_path = os.path.join(proj_dir,"ground_truth.csv")

# Directory Creation 
if not os.path.exists(proj_dir):
   os.makedirs(proj_dir)
