from data_utils import Dataset 
from vectorizer import Vectorizer
from utils_training import Trainer

dataset = Dataset(name='twitter_joy', path='data/vectorizer_joy.pkl', min_length=None, train_lexicon_feat_path="data/joy/train/X_train_joy.txt", 
                    test_lexicon_feat_path="data/joy/dev/X_dev_joy.txt", train_emoji_feat_path="data/joy/train/emoji_emb.txt",
                    test_emoji_feat_path="data/joy/dev/emoji_emb.txt")

params = {
    'vocab_size':dataset.vec.vocab_size,
    'embed_size':200,
    'hidden_size':64,
    'bsize':32,
    'use_lexicons':False,
    'use_emojis':True,
    'lex_feat_length':53,
    'use_attention':True,
    'lexicon_feat_target_dims':10,
    'emoji_feat_target_dims':10,
    'dropout_prob':0.3
}

vocab_size = dataset.vec.vocab_size
embed_size = 200
hidden_size = 128
bsize=32

trainer = Trainer(dataset, params)

trainer.train(n_iters=100, save_on_metric='roc_auc')
