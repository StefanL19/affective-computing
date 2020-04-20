from Models import Model
from tqdm import tqdm
from common.metrics import calc_metrics_classification
import numpy as np

class Trainer() :
    def __init__(self, dataset, params:dict): 
        # pass the model config when refactoring

        self.model = Model(params['vocab_size'], params['embed_size'], params['hidden_size'], params['bsize'], pre_embed=dataset.vec.embeddings, 
        use_lexicons=params['use_lexicons'], use_emojis=params['use_emojis'], lex_feat_length=params['lex_feat_length'], 
        use_attention=params['use_attention'], lexicon_feat_target_dims=params['lexicon_feat_target_dims'], 
        emoji_feat_target_dims=params['emoji_feat_target_dims'], dropout_prob=params['dropout_prob'])

        self.predictions_test_save_path = params['test_predictions_save_path']
        self.lstm_train_context_features_save_path = params['train_context_features_save_path']
        self.lstm_test_context_features_save_path = params['test_context_features_save_path']
        self.metrics = calc_metrics_classification
        self.display_metrics = True
        self.dataset = dataset

    def train(self, n_iters, save_on_metric):
        
        train_data = self.dataset.train_data
        test_data = self.dataset.test_data

        test_metrics = []

        context_vectors_best_train = []
        context_vectors_best_test = []
        embedding_vectors_best_train = []
        embedding_vectors_best_test = []

        best_pearson = -100.
        for i in tqdm(range(n_iters)):
            loss_total_batch, loss_total = self.model.train(train_data.X, train_data.y, train_data.lexicon_feat, train_data.emoji_feat)

            # print("Finished Training Iteration: ", str(i))
            # print("The Average Loss per batch is: ", loss_total_batch)
            # print("The Total Loss is: ", loss_total)

            predictions_test, context_vectors_test, embedding_vectors_test = self.model.evaluate(test_data.X, test_data.lexicon_feat, test_data.emoji_feat)

            predictions_train, context_vectors_train, embedding_vectors_train = self.model.evaluate(train_data.X, train_data.lexicon_feat, train_data.emoji_feat)

            predictions_test = np.array(predictions_test)

            pearson_r = self.metrics(np.array(test_data.y), predictions_test)
            
            if pearson_r[0] >= best_pearson:
                print("New best iteration")
                best_pearson = pearson_r[0]
                print(best_pearson)
                context_vectors_best_train = context_vectors_train
                context_vectors_best_test = context_vectors_test
                
                context_vectors_best_train = np.concatenate(context_vectors_best_train, axis=0)

                context_vectors_best_test = np.concatenate(context_vectors_best_test, axis=0)
                
                embedding_vectors_best_train = np.concatenate(embedding_vectors_train, axis=0)

                embedding_vectors_best_test = np.concatenate(embedding_vectors_test, axis=0)

                np.savetxt(self.predictions_test_save_path, predictions_test)
                np.savetxt(self.lstm_train_context_features_save_path, np.array(context_vectors_best_train))
                np.savetxt(self.lstm_test_context_features_save_path, np.array(context_vectors_best_test))

            test_metrics.append(pearson_r[0])

        return np.max(np.array(test_metrics)), context_vectors_best_train, context_vectors_best_test
