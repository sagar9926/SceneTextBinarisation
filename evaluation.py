import pandas as pd
from thefuzz import fuzz
import numpy as np


def evaluate(gt, extracted):
    score = []
    label = gt.to_numpy()
    out = extracted.to_numpy().reshape(-1)
    for truth, ext in zip(label, out):
        score.append(fuzz.partial_ratio(truth, ext))
    return pd.DataFrame(np.array(score))
