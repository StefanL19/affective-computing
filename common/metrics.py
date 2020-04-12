import numpy as np
from collections import defaultdict
import pandas as pd
import torch
from scipy import stats

def calc_metrics_classification(target, predictions, target_scores=None) :

    if target_scores is not None :
        assert predictions.squeeze(1).shape == target_scores.shape

    if predictions.shape[-1] == 1 :
        predictions = predictions[:, 0]

    return stats.pearsonr(predictions, target)
