from Models import Model
from tqdm import tqdm
from common.metrics import calc_metrics_classification
import numpy as np

class Trainer() :
    def __init__(self, dataset, vocab_size, embed_size, hidden_size, bsize): 
        # pass the model config when refactoring

        self.model = Model(vocab_size, embed_size, hidden_size, bsize, pre_embed=dataset.vec.embeddings, use_lexicons=True, lex_feat_length=53)
        self.metrics = calc_metrics_classification
        self.display_metrics = True
        self.dataset = dataset

    def train(self, n_iters, save_on_metric):
        
        train_data = self.dataset.train_data
        test_data = self.dataset.test_data

        for i in tqdm(range(n_iters)):
            loss_total_batch, loss_total = self.model.train(train_data.X, train_data.y, train_data.lexicon_feat)

            print("Finished Training Iteration: ", str(i))
            print("The Average Loss per batch is: ", loss_total_batch)
            print("The Total Loss is: ", loss_total)

            predictions_test = self.model.evaluate(test_data.X, test_data.lexicon_feat)

            predictions_test = np.array(predictions_test)

            self.metrics(np.array(test_data.y), predictions_test)

            # if self.display_metrics:
            #     print("TEST METRICS: ")
            
            # # Get the value for the metric which should be used for checkpointing
            # metric = test_metrics[save_on_metric]

            # if metric > best_metric :
            #     best_metric = metric

            #     save_model = True

            #     print("Model Saved on ", save_on_metric, metric)

            # else:
            #     save_model = False
            #     print("Model not saved on ", save_on_metric, metric)

        # TODO: implement this function
        #dirname = self.model.save_values(save_model=save_model)

        # TODO: implement model saving
