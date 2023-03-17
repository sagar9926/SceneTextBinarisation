##########################################################################################
######### Parsing labels corresponding to scene text from the .xml files  ################

### Steps to execute the code :
###      1. Unzip the folder with the scene images using the command : !unzip /content/drive/MyDrive/SVT/archive.zip
###      2. execute the data_parse.py code
###      3. 
##########################################################################################
# from google.colab import drive
# drive.mount('/content/drive/')
# !unzip /content/drive/MyDrive/SVT/archive.zip

from bs4 import BeautifulSoup
import re
import pandas as pd
from config import class_label_path,train_xml_path,test_xml_path


## Data Parsing Script 
if __name__ == "__main__" :

  with open(train_xml_path) as f :
    all_tags = f.read()

  image_data_all_tags  = re.findall("<image>(.*?)</image>", all_tags, re.DOTALL)
  df_train = pd.DataFrame([],columns = ['image_path','scene_text'])
  for tag in image_data_all_tags : 
    image_name = " ".join(re.findall("<imageName>(.*?)</imageName>", tag, re.DOTALL))
    text = " ".join(re.findall("<tag>(.*?)</tag>", tag, re.DOTALL))
    row_list = [image_name , text]
    df_train = df_train.append(pd.Series(row_list, index = ['image_path','scene_text']), ignore_index=True)

  with open(test_xml_path) as f :
    all_tags = f.read()

  image_data_all_tags  = re.findall("<image>(.*?)</image>", all_tags, re.DOTALL)
  df_test = pd.DataFrame([],columns = ['image_path','scene_text'])
  for tag in image_data_all_tags : 
    image_name = " ".join(re.findall("<imageName>(.*?)</imageName>", tag, re.DOTALL))
    text = " ".join(re.findall("<tag>(.*?)</tag>", tag, re.DOTALL))
    row_list = [image_name , text]
    df_test = df_test.append(pd.Series(row_list, index = ['image_path','scene_text']), ignore_index=True)

  df_class_labels = pd.concat([df_test,df_train])

  df_class_labels.to_csv(class_labels_path,index = False)
