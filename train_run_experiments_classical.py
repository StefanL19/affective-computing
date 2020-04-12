import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge
from preprocess_data import loadGloveModel
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

def read_data(data_path:str):
    tweets = []
    with open(data_path, "r") as f:
        lines = f.readlines()

        for line in lines:
            tweets.append(line.strip("\n"))
    
    return tweets

def read_labels(data_path:str):
    intensities = []

    with open(data_path, "r") as f:
        lines = f.readlines()

        for line in lines:
            intensities.append(float(line.strip("\n")))
    
    return intensities

def get_xy_emb(data_path:str, labels_path:str, emoji_feats_path:str, lexicon_feats_path:str, glove_model:dict, use_emojis:bool, use_lexicon:bool):
    tweets = read_data(data_path)
    intensities = read_labels(labels_path)

    all_embeddings = []
    for tweet in tqdm(tweets):
        # Tokenize the tweet
        tweet = tweet.split(" ")

        # List to store all token embeddings for this tweet
        tweet_embeddings = []
        for token in tweet:
            if token in glove_model:
                token_embedding = glove_model[token]
                tweet_embeddings.append(token_embedding)

        # Get the average tweet embedding 
        average_embedding = np.mean(np.array(tweet_embeddings), axis=0)

        # If no words were found, then just use the zero embedding
        if average_embedding.shape != (200,):
            average_embedding = np.zeros(shape=(200,))

        all_embeddings.append(average_embedding)
    
    X = np.stack(all_embeddings)

    if use_emojis:
        X = append_emoji_features(X, emoji_feats_path)

    if use_lexicon:
        X = append_lexicon_features(X, lexicon_feats_path)

    y = np.array(intensities)

    print("Shapes X: {}, y: {}".format(X.shape, y.shape))
    
    return X, y

def get_xy_lstm_features(lstm_feats_path:str, labels_path:str, emoji_feats_path:str, lexicon_feats_path:str, use_emojis:bool, use_lexicon:bool):
    X = np.loadtxt(lstm_feats_path)
    intensities = read_labels(labels_path)

    if use_emojis:
        X = append_emoji_features(X, emoji_feats_path)

    if use_lexicon:
        X = append_lexicon_features(X, lexicon_feats_path)

    y = np.array(intensities)

    print("Shapes X: {}, y: {}".format(X.shape, y.shape))

    return X, y


def append_emoji_features(initial_embeddings, emoji_feats_path):
    print("Using emoji features")
    emoji_features = np.loadtxt(emoji_feats_path)

    combined_features = np.concatenate((initial_embeddings, emoji_features), axis=1)

    return combined_features

def append_lexicon_features(initial_embeddings, lexicon_feats_path):
    print("Using lexicon features")
    lexicon_features = np.loadtxt(lexicon_feats_path)

    combined_features = np.concatenate((initial_embeddings, lexicon_features), axis=1)

    return combined_features

def metrics(y_pred, y, print_metrics=False):
    p1 = pearsonr(y_pred, y)[0]
    s1 = spearmanr(y_pred, y)[0]
    ind = np.where(y >= 0.5)
    ydt = np.take(y_pred, ind).reshape(-1)
    ydpt = np.take(y, ind).reshape(-1)
    p2 = pearsonr(ydt, ydpt)[0]
    s2 = spearmanr(ydt, ydpt)[0]
    if print_metrics:
        print("Validation Pearsonr: {}".format(p1))
        print("Validation Spearmanr: {}".format(s1))
        print("Validation Pearsonr >= 0.5: {}".format(p2))
        print("Validation Spearmanr >= 0.5: {}".format(s2))
    return np.array((p1, s1, p2, s2))

def train_only_embeddings(train_data_path:str, train_labels_path:str, train_emojis_path:str, train_lexicon_path:str,
                        dev_data_path:str, dev_labels_path:str, dev_emojis_path:str, dev_lexicon_path:str,
                        glove_model_path:str, model_name:str, use_emojis:bool, use_lexicon:bool, use_cross_validation:str):

    glove_model = loadGloveModel(glove_model_path)
    train_x, train_y = get_xy_emb(train_data_path, train_labels_path, train_emojis_path, train_lexicon_path, glove_model, use_emojis, use_lexicon)
    dev_x, dev_y = get_xy_emb(dev_data_path, dev_labels_path, dev_emojis_path, dev_lexicon_path, glove_model, use_emojis, use_lexicon)
    
    all_metrics = []
    for i in range(0,10):
        if model_name == "RandomForest":
            print("Using random forest model")
            regr = RandomForestRegressor()
        elif model_name == "GradientBoostingRegressor":
            print("Using gradient boosting regressor")
            regr = GradientBoostingRegressor()
        elif model_name == "BaggingRegressor":
            print("Using bagging regressor")
            regr = BaggingRegressor()
        elif model_name == "AdaBoostRegressor":
            print("Using adaboost regressor")
            xg_boost_reg = XGBRegressor(n_estimators=1000, learning_rate=0.1)
            regr = AdaBoostRegressor(base_estimator=xg_boost_reg)

        regr.fit(train_x, train_y)
        y_dev_pred = regr.predict(dev_x)
        metrics_res = metrics(y_dev_pred, dev_y, True)

        all_metrics.append(metrics_res[0])
    
    print("The mean pearson correlation is: ", np.mean(np.array(all_metrics)))


def train_on_lstm_features(train_lstm_feats_path:str, train_labels_path:str, train_emojis_path:str, train_lexicon_path:str,
                        dev_lstm_feats_path:str, dev_labels_path:str, dev_emojis_path:str, dev_lexicon_path:str,
                        glove_model_path:str, model_name:str, use_emojis:bool, use_lexicon:bool):
    train_x, train_y = get_xy_lstm_features(train_lstm_feats_path, train_labels_path, train_emojis_path, train_lexicon_path, use_emojis, use_lexicon)
    dev_x, dev_y = get_xy_lstm_features(dev_lstm_feats_path, dev_labels_path, dev_emojis_path, dev_lexicon_path, use_emojis, use_lexicon)

    
    all_metrics = []
    for i in range(0,5):
        if model_name == "RandomForest":
            print("Using random forest model")
            regr = RandomForestRegressor(n_estimators=30)
        elif model_name == "GradientBoostingRegressor":
            print("Using gradient boosting regressor")
            regr = GradientBoostingRegressor()
        elif model_name == "BaggingRegressor":
            print("Using bagging regressor")
            regr = BaggingRegressor()

        regr.fit(train_x, train_y)
        y_dev_pred = regr.predict(dev_x)
        metrics_res = metrics(y_dev_pred, dev_y, True)

        all_metrics.append(metrics_res[0])
    
    print("The mean pearson correlation is: ", np.mean(np.array(all_metrics)))

    


emotion_type = "fear"
use_lexicon = True
use_emojis = True
use_cross_validation = True

# train_on_lstm_features("data/joy/train/lstm_features.txt", "data/joy/train/joy_train_target.txt", "data/joy/train/emoji_emb.txt", "data/joy/train/X_train_joy.txt",
#                     "data/joy/dev/lstm_features.txt", "data/joy/dev/joy_dev_target.txt", "data/joy/dev/emoji_emb.txt", "data/joy/dev/X_dev_joy.txt",
#                     "data/embeddings/glove.twitter.27B.200d.txt", "RandomForest", use_emojis, use_lexicon)

if emotion_type == "joy":
    print("Results for joy emotion type")
    train_only_embeddings("data/joy/train/joy_train_input.txt", "data/joy/train/joy_train_target.txt", "data/joy/train/emoji_emb.txt", "data/joy/train/X_train_joy.txt",
                        "data/joy/dev/joy_dev_input.txt", "data/joy/dev/joy_dev_target.txt", "data/joy/dev/emoji_emb.txt", "data/joy/dev/X_dev_joy.txt",
                        "data/embeddings/glove.twitter.27B.200d.txt", "AdaBoostRegressor", use_emojis, use_lexicon, use_cross_validation)

elif emotion_type == "anger":
    print("Results for anger emotion type")
    train_only_embeddings("data/anger/train/anger_train_input.txt", "data/anger/train/anger_train_target.txt", "data/anger/train/emoji_emb.txt", "data/anger/train/X_train_anger.txt",
                        "data/anger/dev/anger_dev_input.txt", "data/anger/dev/anger_dev_target.txt", "data/anger/dev/emoji_emb.txt", "data/anger/dev/X_dev_anger.txt",
                        "data/embeddings/glove.twitter.27B.200d.txt", "AdaBoostRegressor", use_emojis, use_lexicon, use_cross_validation)

elif emotion_type == "sadness":
    print("Results for sadness emotion type")
    train_only_embeddings("data/sadness/train/sadness_train_input.txt", "data/sadness/train/sadness_train_target.txt", "data/sadness/train/emoji_emb.txt", "data/sadness/train/X_train_sadness.txt",
                        "data/sadness/dev/sadness_dev_input.txt", "data/sadness/dev/sadness_dev_target.txt", "data/sadness/dev/emoji_emb.txt", "data/sadness/dev/X_dev_sadness.txt",
                        "data/embeddings/glove.twitter.27B.200d.txt", "AdaBoostRegressor", use_emojis, use_lexicon, use_cross_validation)

elif emotion_type == "fear":
    print("Results for fear emotion type")
    train_only_embeddings("data/fear/train/fear_train_input.txt", "data/fear/train/fear_train_target.txt", "data/fear/train/emoji_emb.txt", "data/fear/train/X_train_fear.txt",
                        "data/fear/dev/fear_dev_input.txt", "data/fear/dev/fear_dev_target.txt", "data/fear/dev/emoji_emb.txt", "data/fear/dev/X_dev_fear.txt",
                        "data/embeddings/glove.twitter.27B.200d.txt", "AdaBoostRegressor", use_emojis, use_lexicon, use_cross_validation)
