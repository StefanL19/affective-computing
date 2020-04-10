# Research goals
Check if combining emoji and/or lexicon features together with RNN features (after doing a weighted sum by using attention weights) 
provides any improvement on top of of using either only the lexicon features or only the RNN features for the task of 
predicting emotion intensity in tweets

# Pipeline
## Data preprocessing
Run the script `preprocess_data.py` and change the corresponding variables 
This script uses pre-trained word embeddings to pre-initialize the embedding layer of the RNN model.

## Including lexicon features
The already pre-processed lexicon features files for each emotion should be included in the data directory as text files


