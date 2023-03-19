##evaluation_fn.py
import pandas as pd
from thefuzz import fuzz
import numpy as np

def evaluation_metric(df,ground_truth_col,pred_col,method): 
  """
  This method uses partial ratio raw score is a measure of the strings similarity to
  evalutate the qualiity of the text extracted by the Off the shelf OCR method applied over
  the binary images. 
  """

  df[f'score_{method}'] = df.apply(lambda row : fuzz.partial_ratio(row[ground_truth_col],row[pred_col]),axis = 1)
  
  return df

def evaluate(gt, extracted):
    score = []
    label = gt.to_numpy()
    out = extracted.to_numpy().reshape(-1)
    for truth, ext in zip(label, out):
        score.append(fuzz.partial_ratio(truth, ext))
    return pd.DataFrame(np.array(score))

  
