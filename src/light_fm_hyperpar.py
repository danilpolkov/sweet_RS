import numpy as np
import itertools
from datetime import datetime
from lightfm import LightFM
from scipy.sparse import coo_matrix
import ml_metrics as metrics


def sample_hyperparameters():
    """
    Yield possible hyperparameter choices.
    """

    while True:
        yield {
            "no_components": np.random.randint(16, 64),
            "learning_schedule": np.random.choice(["adagrad", "adadelta"]),
            "loss": np.random.choice(["bpr", "warp", "warp-kos"]),
            "learning_rate": np.random.exponential(0.05),
            "num_epochs": np.random.randint(5, 30),
        }


def random_search(train, user_hist, correct: dict, items_to_predict, num_samples: int = 20, num_threads: int = -1):
    """
    Sample random hyperparameters, fit a LightFM model, and evaluate it
    on the test set.
    Parameters
    ----------
    train: np.float32 coo_matrix
        Training data.
    correct: dict
        dict with keys as item and val is max score 
    num_samples: int, optional
        Number of hyperparameter choices to evaluate.
    Returns
    ----------
    generator of (auc_score, hyperparameter dict, fitted model)
    """
    best_score = -1
    best_params = {}
    for hyperparams in itertools.islice(sample_hyperparameters(), num_samples):
        start = datetime.now()
        print('hyperparams set:', hyperparams)
        num_epochs = hyperparams.pop("num_epochs")

        model = LightFM(**hyperparams)
        model.fit(train, epochs=num_epochs, num_threads=num_threads)

        recoms = {}
        num_to_recom = 5
        for user in correct.keys():
            items_to_score = list(items_to_predict.difference(user_hist[user]))
            predict = model.predict(
                user, items_to_predict, num_threads=num_threads)
            top_recoms_id = sorted(range(len(predict)),
                                   key=lambda i: predict[i])[-num_to_recom:]
            top_recoms_id.reverse()
            recoms[user_decode[user]] = [item_decode[items_to_predict[i]]
                                         for i in top_recoms_id]
    
        score = metrics.mapk(list(recoms.values()), list(correct_1.values()), 5)
        print(score)

        hyperparams["num_epochs"] = num_epochs

        end = datetime.now()

        yield (score, hyperparams, model, end - start)
        
        
def new_func():
    print('ok')