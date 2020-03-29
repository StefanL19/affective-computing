import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from math import ceil
from tqdm import tqdm
from torchtext.vocab import pretrained_aliases
import spacy, re

SOS = "<SOS>"
EOS = "<EOS>"
PAD = "<0>"
UNK = "<UNK>"

class Vectorizer:
    def __init__(self, num_words=None, min_df=None):
        self.embeddings = None
        self.word_dim = 200
        self.num_words = num_words
        self.min_df = min_df
    
    def process_to_docs(self, texts):
        '''
            Removes the end of line token from every sentence in the list  
        '''
        docs = [t.replace("\n", " ").strip() for t in texts]
        return docs

    def process_to_sentences(self, texts):
        '''
            Processes a text file to a set of documents by splitting it by the newline token
        '''
        docs = [t.split("\n") for t in texts]
        return docs

    def tokenizer(self, text):
        '''
            Tokenizes a sentence
        '''
        return text.split(" ")

    def fit(self, texts):
        '''
            Creates word2idx and idx to word dictionaries from the passed texts
        '''
        if self.min_df is not None:
            self.cvec = CountVectorizer(tokenizer=self.tokenizer, min_df=self.min_df, lowercase=False)
        else:
            self.cvec = CountVectorizer(tokenizer=self.tokenizer, lowercase=False)
        
        bow = self.cvec.fit_transform(texts)

        self.word2idx = self.cvec.vocabulary_

        for word in self.cvec.vocabulary_:
            self.word2idx[word] += 4

        self.word2idx[PAD] = 0
        self.word2idx[UNK] = 1
        self.word2idx[SOS] = 2
        self.word2idx[EOS] = 3

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

        self.cvec.stop_words_ = None

    def add_word(self, word):
        if word not in self.word2idx:
            idx = max(self.word2idx.values()) + 1
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            self.vocab_size += 1

    def fit_docs(self, texts):
        '''
            Fits the input texts. The texts should be a string where each new sentence is separated by a newline token
        '''
        docs = self.process_to_docs(texts)
        self.fit(docs)

    def convert_to_sequence(self, texts):
        texts_tokenized = map(self.tokenizer, texts)
        texts_tokenized = map(lambda s: [SOS] + [UNK if word not in self.word2idx else word for word in s] + [EOS], texts_tokenized)
        texts_tokenized = list(texts_tokenized)
        sequences = map(lambda s: [int(self.word2idx[word]) for word in s], texts_tokenized)
        return list(sequences)

    def texts_to_sequences(self, texts):
        unpad_X = self.convert_to_sequence(texts)
        return unpad_X

    def extract_embeddings(self, model, vector_size):
        '''
            Extracts the embeddings from a pre-trained embeddings model for each one of the words in the corpora
        '''
        # Get the dimensionality for rows and cols
        self.word_dim, self.vocab_size = vector_size, len(self.word2idx)
        
        # Pre-Initialize the embeddings matrix with zeros
        self.embeddings = np.zeros([self.vocab_size, self.word_dim])
        
        # Counter for the words for which pre-trained embeddings were not found
        in_pre = 0
        for i, word in sorted(self.idx2word.items()):
            if word in model:
                self.embeddings[i] = model[word]
                in_pre += 1
            else:
                self.embeddings[i] = np.random.randn(self.word_dim)

        # Zero embedding for the zero word used for padding
        self.embeddings[0] = np.zeros(self.word_dim)

        print("Found " + str(in_pre) + " words in model out of " + str(len(self.idx2word)))

        return self.embeddings

    def extract_embeddings_from_torchtext(self, model):
        '''
            Extracts the embeddings from a pre-trained embeddings model for each one of the words in the corpora by using the torchtext api
        '''
        vectors = pretrained_aliases[model](cache='../.vector_cache')
        self.word_dim = vectors.dim
        self.embeddings = np.zeros((len(self.idx2word), self.word_dim))
        in_pre = 0
        for i, word in self.idx2word.items():
            if word in vectors.stoi : 
                in_pre += 1                
                self.embeddings[i] = vectors[word].numpy()
                # otherwise the zero vector will be there either way

        # Zero embedding for the zero word
        self.embeddings[0] = np.zeros(self.word_dim)
        print("Found " + str(in_pre) + " words in model out of " + str(len(self.idx2word)))
        return self.embeddings

    def get_seq_for_docs(self, texts):

        # Get the documents
        docs = self.process_to_docs(texts)  # D

        # Get a tokenized sequence for each one of the documents
        seq = self.texts_to_sequences(docs)  # D x W

        return seq

    def get_seq_for_sents(self, texts):
        sents = self.process_to_sentences(texts)  # (D x S)
        seqs = []
        for d in tqdm(sents):
            seqs.append(self.texts_to_sequences(d))

        return seqs

    def map2words(self, sent):
        return [self.idx2word[x] for x in sent]

    def map2words_shift(self, sent):
        return [self.idx2word[x + 4] for x in sent]

    def map2idxs(self, words):
        return [self.word2idx[x] if x in self.word2idx else self.word2idx[UNK] for x in words]

    def add_frequencies(self, X):
        freq = np.zeros((self.vocab_size,))
        for x in X:
            for w in x:
                freq[w] += 1
        freq = freq / np.sum(freq)
        self.freq = freq
