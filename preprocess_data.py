from vectorizer import Vectorizer
import pandas as pd
import emoji
import numpy as np

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

def replace_emojis(doc):
    doc = emoji.demojize(doc)
    return doc

if __name__ == "__main__":
    train_texts_data_path = "data/joy/train/joy_train_input.txt"
    train_scores_data_path = "data/joy/train/joy_train_target.txt" 
    dev_texts_data_path = "data/joy/dev/joy_dev_input.txt"
    dev_scores_data_path = "data/joy/dev/joy_dev_target.txt"

    with open(train_texts_data_path) as f_train:
        train_texts_content = f_train.readlines()

    with open(dev_texts_data_path) as f_dev:
        dev_texts_content = f_dev.readlines()

    with open(train_scores_data_path) as f_train_sc:
        train_scores_content = f_train_sc.readlines()

    with open(dev_scores_data_path) as f_dev_sc:
        dev_scores_content = f_dev_sc.readlines()

    vec = Vectorizer(min_df=1)

    train_scores = vec.process_to_docs(train_scores_content)
    train_scores = [float(score) for score in train_scores]

    dev_scores = vec.process_to_docs(dev_scores_content)
    dev_scores = [float(score) for score in dev_scores]

    train_texts = vec.process_to_docs(train_texts_content)
    dev_texts = vec.process_to_docs(dev_texts_content)

    # Replace the emojis with corresponding strings
    train_texts = [replace_emojis(t) for t in train_texts]
    dev_texts = [replace_emojis(t) for t in dev_texts]

    vec.fit(train_texts)

    print("Vocabulary size : ", vec.vocab_size)

    vec.seq_text = {}
    vec.label = {}

    vec.seq_text["train"] = vec.get_seq_for_docs(train_texts)
    vec.seq_text["dev"] = vec.get_seq_for_docs(dev_texts)
    vec.label["train"] = train_scores
    vec.label["dev"] = dev_scores

    print("The size of the training set is: ", len(vec.seq_text["train"]))
    print("The size of the validation set is: ", len(vec.seq_text["dev"]))

    pre_trained_embeddings = loadGloveModel("data/embeddings/glove.twitter.27B.200d.txt")

    vec.extract_embeddings(pre_trained_embeddings, 200)

    import pickle, os
    pickle.dump(vec, open("data/vectorizer_joy.pkl", 'wb'))
