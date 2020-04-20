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
from sklearn.model_selection import cross_val_score
from error_analysis import analyse_errors
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.neural_network import MLPRegressor

def pearson_r_loss(target, predictions):
    print(target.shape)
    print(predictions.shape)
    pearson_r_res = pearsonr(predictions, target)
    print(pearson_r_res)
    return pearson_r_res[0]

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
    print(X.shape)
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
    print(lexicon_features.shape)

    combined_features = np.concatenate((initial_embeddings, lexicon_features), axis=1)

    return combined_features

def metrics(y_pred, y, print_metrics=False):
    p1 = pearsonr(y_pred, y)[0]
    s1 = spearmanr(y_pred, y)[0]

    if print_metrics:
        print("Validation Pearsonr: {}".format(p1))
        print("Validation Spearmanr: {}".format(s1))

    return np.array((p1, s1, 0, 0))

def train_only_embeddings(train_data_path:str, train_labels_path:str, train_emojis_path:str, train_lexicon_path:str,
                        dev_data_path:str, dev_labels_path:str, dev_emojis_path:str, dev_lexicon_path:str,
                        glove_model_path:str, model_name:str, use_emojis:bool, use_lexicon:bool, test_predictions_save_path:str):

    glove_model = loadGloveModel(glove_model_path)
    train_x, train_y = get_xy_emb(train_data_path, train_labels_path, train_emojis_path, train_lexicon_path, glove_model, use_emojis, use_lexicon)
    dev_x, dev_y = get_xy_emb(dev_data_path, dev_labels_path, dev_emojis_path, dev_lexicon_path, glove_model, use_emojis, use_lexicon)

    all_metrics = []
    for i in range(0,5):
        if model_name == "RandomForest":
            print("Using random forest model")
            regr = RandomForestRegressor(n_estimators=250)
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
        elif model_name == "SVM":
            print("Using SVM classifier")
            regr = MLPRegressor()

        regr.fit(train_x, train_y)
        y_dev_pred = regr.predict(dev_x)
        np.savetxt(test_predictions_save_path, y_dev_pred)

        metrics_res = metrics(y_dev_pred, dev_y, True)

        all_metrics.append(metrics_res[0])
    
    final_predictions = regr.predict(dev_x)
    print("The mean pearson correlation is: ", np.mean(np.array(all_metrics)))

    return final_predictions

def search_params_embeddings(train_data_path:str, train_labels_path:str, train_emojis_path:str, train_lexicon_path:str,
                        dev_data_path:str, dev_labels_path:str, dev_emojis_path:str, dev_lexicon_path:str,
                        glove_model_path:str, model_name:str, use_emojis:bool, use_lexicon:bool):

    glove_model = loadGloveModel(glove_model_path)
    train_x, train_y = get_xy_emb(train_data_path, train_labels_path, train_emojis_path, train_lexicon_path, glove_model, use_emojis, use_lexicon)
    dev_x, dev_y = get_xy_emb(dev_data_path, dev_labels_path, dev_emojis_path, dev_lexicon_path, glove_model, use_emojis, use_lexicon)

    X = np.concatenate((train_x, dev_x))
    y = np.concatenate((train_y, dev_y))

    scorer = make_scorer(pearson_r_loss, greater_is_better=True)

    tuned_parameters = {
        "n_estimators":[50, 100, 150, 200, 250]
    }

    print("Loaded data")
    grid = GridSearchCV(RandomForestRegressor(), param_grid=tuned_parameters, scoring=scorer, cv=10)

    grid.fit(X, y)

    print(grid.best_params_)

def train_on_lstm_features(train_lstm_feats_path:str, train_labels_path:str, train_emojis_path:str, train_lexicon_path:str,
                        dev_lstm_feats_path:str, dev_labels_path:str, dev_emojis_path:str, dev_lexicon_path:str,
                        glove_model_path:str, model_name:str, use_emojis:bool, use_lexicon:bool, predictions_save_path:str):
    train_x, train_y = get_xy_lstm_features(train_lstm_feats_path, train_labels_path, train_emojis_path, train_lexicon_path, use_emojis, use_lexicon)
    dev_x, dev_y = get_xy_lstm_features(dev_lstm_feats_path, dev_labels_path, dev_emojis_path, dev_lexicon_path, use_emojis, use_lexicon)

    
    all_metrics = []

    for i in range(0,5):
        if model_name == "RandomForest":
            print("Using random forest model")
            regr = RandomForestRegressor(n_estimators=100)
        elif model_name == "GradientBoostingRegressor":
            print("Using gradient boosting regressor")
            regr = GradientBoostingRegressor()
        elif model_name == "BaggingRegressor":
            print("Using bagging regressor")
            regr = BaggingRegressor()

        regr.fit(train_x, train_y)
        y_dev_pred = regr.predict(dev_x)

        np.savetxt(predictions_save_path, y_dev_pred)

        metrics_res = metrics(y_dev_pred, dev_y, True)

        all_metrics.append(metrics_res[0])
    
    print("The mean pearson correlation is: ", np.mean(np.array(all_metrics)))

def test_ensemble(lstm_predictions_path:str, random_forest_predictions_path:str):
    lstm_preds = np.loadtxt(lstm_predictions_path)
    classical_predictions = np.loadtxt(random_forest_predictions_path) 

    
    gt = np.array(read_labels("data/joy/dev/joy_dev_target.txt"))
    ensemble_preds = (lstm_preds + classical_predictions) / 2.
    
    print("Ensemble model metrics: ")
    metrics(ensemble_preds, gt, True)

    df_predictions = analyse_errors("data/joy/dev/joy-ratings-0to1.dev.target.txt", gt, lstm_preds)
    

    outliers = df_predictions[(df_predictions['ground_truth'] <= 0.30) | (df_predictions['ground_truth'] >= 0.70)]
    in_range = df_predictions[(df_predictions['ground_truth'] > 0.30) & (df_predictions['ground_truth'] < 0.70)]
    

    outliers_predictions = np.array(list(outliers["model_predictions"]))
    outliers_ground_truth = np.array(list(outliers["ground_truth"]))

    in_range_predictions = np.array(list(in_range["model_predictions"]))
    in_range_ground_truth = np.array(list(in_range["ground_truth"]))

    print("Mean absolute error outliers:")
    print((np.abs(outliers_predictions - outliers_ground_truth)).mean(axis=0))

    print("Mean absolute error inliers:")    
    print((np.abs(in_range_predictions - in_range_ground_truth)).mean(axis=0))

    print("Percent outliers:")
    print(outliers_predictions.shape[0] / (outliers_predictions.shape[0] + in_range_predictions.shape[0]))

    print("Percent inliers:")
    print(in_range_predictions.shape[0] / (outliers_predictions.shape[0] + in_range_predictions.shape[0]))

def explore_distributions(train_target_path:str, test_target_path:str):
    train_target = np.array(read_labels(train_target_path))
    test_target = np.array(read_labels(test_target_path))

    print("Train distribution:")
    print(np.mean(train_target))
    print(np.std(train_target))

    print("Test distribution:")
    print(np.mean(test_target))
    print(np.std(test_target))

# search_params_embeddings("data/joy/train/joy_train_input.txt", "data/joy/train/joy_train_target.txt", "data/joy/train/emoji_emb.txt", "data/joy/train/X_train_joy.txt",
#                          "data/joy/dev/joy_dev_input.txt", "data/joy/dev/joy_dev_target.txt", "data/joy/dev/emoji_emb.txt", "data/joy/dev/X_dev_joy.txt",
#                          "data/embeddings/glove.twitter.27B.200d.txt", "RandomForest", True, True)



task = "train_random_forest"


if task == "train_random_forest":
    emotion_type = "joy"
    use_lexicon = True
    use_emojis = True

    if emotion_type == "joy":
        print("Results for joy emotion type")
        dev_preds = train_only_embeddings("data/joy/train/joy_train_input.txt", "data/joy/train/joy_train_target.txt", "data/joy/train/emoji_emb.txt", "data/joy/train/X_train_joy_optim.txt",
                            "data/joy/dev/joy_dev_input.txt", "data/joy/dev/joy_dev_target.txt", "data/joy/dev/emoji_emb.txt", "data/joy/dev/X_dev_joy_optim.txt",
                            "data/embeddings/glove.twitter.27B.200d.txt", "RandomForest", use_emojis, use_lexicon, "data/predictions/random_forest_joy")

    elif emotion_type == "anger":
        print("Results for anger emotion type")
        train_only_embeddings("data/anger/train/anger_train_input.txt", "data/anger/train/anger_train_target.txt", "data/anger/train/emoji_emb.txt", "data/anger/train/X_train_anger_optim.txt",
                            "data/anger/dev/anger_dev_input.txt", "data/anger/dev/anger_dev_target.txt", "data/anger/dev/emoji_emb.txt", "data/anger/dev/X_dev_anger_optim.txt",
                            "data/embeddings/glove.twitter.27B.200d.txt", "RandomForest", use_emojis, use_lexicon, "data/predictions/random_forest_anger")

    elif emotion_type == "sadness":
        print("Results for sadness emotion type")
        train_only_embeddings("data/sadness/train/sadness_train_input.txt", "data/sadness/train/sadness_train_target.txt", "data/sadness/train/emoji_emb.txt", "data/sadness/train/X_train_sadness_optim.txt",
                            "data/sadness/dev/sadness_dev_input.txt", "data/sadness/dev/sadness_dev_target.txt", "data/sadness/dev/emoji_emb.txt", "data/sadness/dev/X_dev_sadness_optim.txt",
                            "data/embeddings/glove.twitter.27B.200d.txt", "RandomForest", use_emojis, use_lexicon, "data/predictions/random_forest_sadness")

    elif emotion_type == "fear":
        print("Results for fear emotion type")
        train_only_embeddings("data/fear/train/fear_train_input.txt", "data/fear/train/fear_train_target.txt", "data/fear/train/emoji_emb.txt", "data/fear/train/X_train_fear_optim.txt",
                            "data/fear/dev/fear_dev_input.txt", "data/fear/dev/fear_dev_target.txt", "data/fear/dev/emoji_emb.txt", "data/fear/dev/X_dev_fear_optim.txt",
                            "data/embeddings/glove.twitter.27B.200d.txt", "RandomForest", use_emojis, use_lexicon, "data/predictions/random_forest_fear")

elif task == "train_random_forest_on_lstm_features":
    train_on_lstm_features("data/anger/train/lstm_features.txt", "data/fear/train/fear_train_target.txt", "data/fear/train/emoji_emb.txt", "data/fear/train/X_train_fear_optim.txt",
                        "data/anger/dev/lstm_features.txt", "data/fear/dev/fear_dev_target.txt", "data/fear/dev/emoji_emb.txt", "data/fear/dev/X_dev_fear_optim.txt",
                        "data/embeddings/glove.twitter.27B.200d.txt", "RandomForest", use_emojis, use_lexicon, "data/predictions/random_forest_lstm_anger.txt")

elif task == "test_ensemble":
    lstm_predictions = "data/predictions/lstm_joy.txt"
    random_forest_predictions = "data/predictions/random_forest_joy.txt"

    test_ensemble(lstm_predictions, random_forest_predictions)

elif task == "explore_distributions":
    train_labels_path = "data/joy/train/joy_train_target.txt"
    test_labels_path = "data/joy/dev/joy_dev_target.txt"
    explore_distributions(train_labels_path, test_labels_path)

else:
    print("Unknown task")
