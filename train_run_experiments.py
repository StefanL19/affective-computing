from data_utils import Dataset 
from vectorizer import Vectorizer
from utils_training import Trainer

dataset = Dataset(name='twitter_joy', path='data/vectorizer_sadness.pkl', min_length=None)

vocab_size = dataset.vec.vocab_size
embed_size = 100
hidden_size = 128
bsize=32

trainer = Trainer(dataset, vocab_size, embed_size, hidden_size, bsize)

trainer.train(n_iters=70, save_on_metric='roc_auc')
