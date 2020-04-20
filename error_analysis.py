import pandas as pd
import numpy as np 
import operator

def analyse_errors(validation_set_path, gs_intensities, model_predictions):
    df = pd.read_csv(validation_set_path, header=None, sep='\t')
    tweets = df[1]

    prediction_differences = list(np.abs(model_predictions - gs_intensities))
    tweets = list(tweets)
    gt_predictions = list(gs_intensities)
    model_predictions = list(model_predictions) 


    df_predictions = pd.DataFrame({"tweets":tweets, "prediction_differences":prediction_differences, 
    "ground_truth":gt_predictions, "model_predictions":model_predictions})

    df_predictions = df_predictions.sort_values(by=["prediction_differences"])

    return df_predictions
