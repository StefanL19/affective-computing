import pandas as pd
import emoji
import re
from preprocess_data import loadGloveModel
import numpy as np

def extract_emojis_doc(a_list:list):
    emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
    r = re.compile('|'.join(re.escape(p) for p in emojis_list))
    aux=[' '.join(r.findall(s)) for s in a_list]
    aux = [a for a in aux if a != '']
    return(aux)

def get_emoji_average_emb(text_emojis:list, emoji_embeddings:dict):
    doc_emb = np.zeros(shape=(300,))
    found_emojis = 0

    for e in text_emojis:
        if e in emoji_embeddings:
            doc_emb += emoji_embeddings[e]
            found_emojis += 1
        
        if found_emojis == 0:
            found_emojis = 1
        
        doc_emb = doc_emb/found_emojis
    
    return doc_emb


def extract_emojis_dataset(dataset_path:str, emoji2vec:dict):
    df = pd.read_csv(dataset_path, header=None, sep='\t')
    tweets = df[1]

    emoji_embeddings = []
    for t in tweets:
        text_emojis = extract_emojis_doc(t)
        emoji_emb = get_emoji_average_emb(text_emojis, emoji2vec)
        emoji_embeddings.append(emoji_emb)

    return np.array(emoji_embeddings)

train_path = "data/joy/train/joy-ratings-0to1.train.txt"
dev_path = "data/joy/dev/joy-ratings-0to1.dev.target.txt"

train_emoji_save_path = "data/joy/train/emoji_emb.txt"
dev_emoji_save_path = "data/joy/dev/emoji_emb.txt"

emoji2vec_embeddings = loadGloveModel("data/embeddings/emoji2vec.txt")

train_emoji_embeddings = extract_emojis_dataset(train_path, emoji2vec_embeddings)
dev_emoji_embeddings = extract_emojis_dataset(dev_path, emoji2vec_embeddings)

np.savetxt(train_emoji_save_path, train_emoji_embeddings)
np.savetxt(dev_emoji_save_path, dev_emoji_embeddings)
