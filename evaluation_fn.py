##evaluation_fn.py
import pandas as pd
from thefuzz import fuzz

def evaluation_metric(df,ground_truth_col,pred_col,method): 
  """
  This method uses partial ratio raw score is a measure of the strings similarity to
  evalutate the qualiity of the text extracted by the Off the shelf OCR method applied over
  the binary images. 
  """

  df_merge[f'score_{method}'] = df_merge.apply(lambda row : fuzz.partial_ratio(row[ground_truth_col],row[pred_col]),axis = 1)
  
  return df

  