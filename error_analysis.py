import pandas as pd
import numpy as np 
import operator

def analyse_errors(validation_set_path, gs_intensities, model_predictions):
    df = pd.read_csv(validation_set_path, header=None, sep='\t')
    tweets = df[1]

    prediction_differences = np.abs(model_predictions - gs_intensities)

    tweet_diffs = dict(zip(tweets, list(prediction_differences)))

    tweet_diffs = sorted(tweet_diffs.items(), key=operator.itemgetter(1))

    top_preds = tweet_diffs[:5]
    bottom_preds = tweet_diffs[-5:]
