# Research goals
Check if combining emoji and/or lexicon features together with RNN features (after doing a weighted sum by using attention weights) 
provides any improvement on top of of using either only the lexicon features or only the RNN features for the task of 
predicting emotion intensity in tweets

# Pipeline
## Data preprocessing
Use the script `preprocess_data.py` 
In order to create an save a vectorizer which can be later reused in training the LSTM model run the script after changing the following parameters in the `main` function:
`train_texts_data_path`: The path to the training set
`train_scores_data_path`: The path to the ground truth scores for the training set
`dev_texts_data_path`: The path to the test set
`dev_scores_data_path`: The path to the ground truth scores for the test set
`save_vectorizer_path`: The path where the vectorizer should be saved

It will be easier to use the repository if you stick to the path conventions which were used to name the files.

## Train LSTM model
Use the script `train_run_experiments.py`
In order to train an LSTM model on one of the emotion types and save the its, predictions, and cotext features on the train and test set run the script
and change the correspond parameters in the `run_training` function. This is the description of the parameters which should be change more often:

When instantiating the `Dataset` class:
`Dataset.path`:path to the vectorizer file which should be used for training the model
`Dataset.train_lexicon_feat_path`: path to the extracted lexicon features for the training set (check the Extracted Features section for more details obtaining the paths to the corresponding emotion types)
`Dataset.test_lexicon_feat_path`: path to the extracted lexicon features for the test set (check the Extracted Features section for more details obtaining the paths to the corresponding emotion types)
`Dataset.train_emoji_feat_path`: path to the extracted emoji features for the train set (check the Extracted Features section for more details obtaining the paths to the corresponding emotion types)
`Dataset.test_emoji_feat_path`: path to the extracted emoji features for the test set (check the Extracted Features section for more details obtaining the paths to the corresponding emotion types)

In the params dictionary:
`test_predictions_save_path`:Path for saving the test set predictions of the best model (based on test set pearson-r correlation)
`train_context_features_save_path`: Path for saving the training set context-level features of the best model (based on test set pearson-r correlation) 
`test_context_features_save_path`: Path for saving the test set context-level features of the best model (based on test set pearson-r correlation) 
`use_lexicons`: Whether the lexicon features should be used for training the model
`use_emojis`: Whether the emoji features should be used for training the model

## Train a Random Forest model, test ensemble model, or explore datasets
Use the script `train_run_experiments_classical.py`
* To train a Random Forest model on one of the emotion types set the `task` variable to `train_random_forest`. Further, change the `emotion_type` parameter and set it to one of `fear`, `anger`, `sadness`, or `joy` and set `use_lexicon` to `True` if you want to use the affective lexicon features and 
`use_emojis` to `True` if you want to use the emoji features. The paths to the data files are set in the file. 

* To explore the dataset distribution files set the `task` variable to `explore_distributions` and pass the paths to the files containing the 
ground truth scores for the train and the test sets. 

* To test an ensemble model set the `task` variable to `test_ensemble` and pass the paths to the predictions of the `LSTM` and `Random Forest` models on the test set.

* To train a `Random Forest` model by utilizing `LSTM` context features instead of the average word embeddings set the `task` variable to `train_random_forest_on_lstm_features`. Further, find and change the name of the emotion type in all variables passed to the `train_on_lstm_features` function.
