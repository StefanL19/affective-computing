from data_utils import Dataset 
from vectorizer import Vectorizer
from utils_training import Trainer

dataset = Dataset(name='twitter_joy', path='data/vectorizer_joy.pkl', min_length=None, train_lexicon_feat_path="data/anger/train/X_train_anger.txt", 
                    test_lexicon_feat_path="data/anger/dev/X_dev_anger.txt")

vocab_size = dataset.vec.vocab_size
embed_size = 200
hidden_size = 128
bsize=32

trainer = Trainer(dataset, vocab_size, embed_size, hidden_size, bsize)

trainer.train(n_iters=100, save_on_metric='roc_auc')
