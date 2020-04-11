from data_utils import Dataset 
from vectorizer import Vectorizer
from utils_training import Trainer
from hyperopt import fmin, tpe, hp, Trials

def objective(space):

    dataset = Dataset(name='twitter_fear', path='data/vectorizer_fear.pkl', min_length=None, train_lexicon_feat_path="data/fear/train/X_train_fear.txt", 
                        test_lexicon_feat_path="data/fear/dev/X_dev_fear.txt", train_emoji_feat_path="data/fear/train/emoji_emb.txt",
                        test_emoji_feat_path="data/fear/dev/emoji_emb.txt")

    params = {
        'vocab_size':dataset.vec.vocab_size,
        'embed_size':200,
        'hidden_size':space['hidden_size'],
        'bsize':32,
        'use_lexicons':space['use_lexicons'],
        'use_emojis':space['use_emojis'],
        'lex_feat_length':53,
        'use_attention':True,
        'lexicon_feat_target_dims':space['lexicon_feat_target_dims'],
        'emoji_feat_target_dims':space['emoji_feat_target_dims'],
        'dropout_prob':space['dropout_prob']
    }

    trainer = Trainer(dataset, params)


    best_result = trainer.train(n_iters=70, save_on_metric='roc_auc')
    
    space['result'] = best_result

    with open('data/hyperopt_fear_results.txt', 'a') as f:
        f.write(str(space))
        f.write('\n')
    
    return (1. - best_result)

space = {
    'hidden_size': hp.choice('hidden_sieze', [64, 128, 32]),
    'dropout_prob': hp.choice('dropout_prob', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    'use_lexicons': hp.choice('use_lexicons', [True, False]),
    'use_emojis': hp.choice('use_emojis', [True, False]),
    'lexicon_feat_target_dims': hp.randint('lexicon_feat_target_dims', 50),
    'emoji_feat_target_dims': hp.randint('emoji_feat_target_dims', 50)
}

trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials
)

print(trials.results)


