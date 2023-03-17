##########################################################################################
######### Parsing labels corresponding to scene text from the .xml files  ################
##########################################################################################

from bs4 import BeautifulSoup
import re
import pandas as pd
from google.colab import drive

drive.mount('/content/drive/')

!unzip /content/drive/MyDrive/SVT/archive.zip

class_label_path = "/content/drive/MyDrive/SVT/ground_truth.csv"

## Data Parsing Script 
if __name__ == "__main__" :

  with open('/content/train.xml') as f :
    all_tags = f.read()

  image_data_all_tags  = re.findall("<image>(.*?)</image>", all_tags, re.DOTALL)
  df_train = pd.DataFrame([],columns = ['image_path','scene_text'])
  for tag in image_data_all_tags : 
    image_name = " ".join(re.findall("<imageName>(.*?)</imageName>", tag, re.DOTALL))
    text = " ".join(re.findall("<tag>(.*?)</tag>", tag, re.DOTALL))
    row_list = [image_name , text]
    df_train = df_train.append(pd.Series(row_list, index = ['image_path','scene_text']), ignore_index=True)

  with open('/content/test.xml') as f :
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
