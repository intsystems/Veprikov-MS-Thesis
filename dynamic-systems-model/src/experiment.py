import pandas as pd
from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb

from sklearn.linear_model import Ridge, RidgeCV, LinearRegression, SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tqdm import tqdm

import random

import ames_housing

import warnings
warnings.filterwarnings("ignore")

def init_random_state(seed):
    np.random.seed(int(seed))
    random.seed = int(seed)

def gbr_model(**kwargs):
    """
    Creates a boosted trees regressor trained with a gradient descent method.

    Uses `GradientBoostingRegressor` from `scikit-learn`

    :param kwargs: parameters for `GradientBoostingRegressor`
    :return: an instance of regressor
    """
    return GradientBoostingRegressor(**kwargs)


def ridgecv_model(**kwargs):
    """
    Creates a pipeline of a data scaler and a ridge regression with cross validation.
    Scaler transforms data to zero mean and unit variance.
    Hyperparameters or the regression are tuned via cross-validation

    Uses `StandardScaler` and `RidgeCV` from `scikit-learn`
    :param kwargs: parameters for `RidgeCV`
    :return: an instance of `Pipeline`
    """

    return Pipeline([['scaler', StandardScaler()], ['ridgecv', RidgeCV(**kwargs)]])

def ridge_model(**kwargs):
    """
    Creates a pipeline of a data scaler and a ridge regression without cross validation.
    Scaler transforms data to zero mean and unit variance.

    Uses `StandardScaler` and `Ridge` from `scikit-learn`
    :param kwargs: parameters for `Ridge`
    :return: an instance of `Pipeline`
    """

    return Pipeline([['scaler', StandardScaler()], ['ridge', Ridge(**kwargs)]])

def linear_model(**kwargs):
    """
    Creates a pipeline of a data scaler and a Linear regression.
    Scaler transforms data to zero mean and unit variance.

    Uses `StandardScaler` and `LinearRegression` from `scikit-learn`
    :param kwargs: parameters for `LinearRegression`
    :return: an instance of `Pipeline`
    """

    return Pipeline([['scaler', StandardScaler()], ['linear', LinearRegression(**kwargs)]])

def sgd_model(**kwargs):
    """
    Creates a pipeline of a data scaler and a SGD regression.
    Scaler transforms data to zero mean and unit variance.

    Uses `StandardScaler` and `SGDRegressor` from `scikit-learn`
    :param kwargs: parameters for `SGDRegressor`
    :return: an instance of `Pipeline`
    """

    return Pipeline([['scaler', StandardScaler()], ['sgd', SGDRegressor(**kwargs)]])

def get_synthetic_dataset(noise, seed):
    """
    Return a regression syntetic dataset from `scikit-learn`
    :return: X, y
    """
    # get dataset
    X, y = datasets.make_regression(n_samples=2000, n_targets=1, 
                                    noise=noise, n_features=10, random_state=seed)
    return X, y

def get_california_dataset(data_len=None):
    """
    Return the california housing dataset from `scikit-learn`
    :return: X, y
    """
    housing = datasets.fetch_california_housing()
    return housing.data[:data_len], housing.target[:data_len]

def get_ames_dataset(data_len=None):
    """
    Return the Ames housing dataset from:
    http://jse.amstat.org/v19n3/decock/AmesHousing.txt
    :return: X, y
    """
    df = ames_housing.load_ames_housing()
    df = ames_housing.clean_data(df)
    X = df.drop(columns=['SalePrice']).select_dtypes(include=['int64', 'float64'])
    y = np.array(df['SalePrice'], dtype=np.float64)
    return np.array(X)[:data_len], y[:data_len]


class HiddenLoopExperiment:
    """
    The main experiment for hidden loops paper
    See details in the paper.

    In short.

    Creates a feedback loop on a regression problem (e.g. Boston housing).
    Some of the model predictions are adhered to by users and fed back into the model as training data.
    Users add a normally distributed noise to the log of the target variable (price).
    Uses a sliding window to retrain the model on new data.

    """


    # params = {'loss':'huber', 'n_estimators': 1000, 'max_depth': 5, 'max_features': 1.0,
    #          'learning_rate': 0.01, 'random_state':0, 'subsample':0.5}
    default_params = {'loss':'ls', 'n_estimators': 50, 
                      'max_depth': 3, 'max_features': 1.0,
                      'learning_rate': 0.5, 'random_state':0,
                      'subsample':0.75}

    default_state = {
        'r2': 'R^2, dynamic data',
        'mae': 'MAE, dynamic_data',
        'r2_orig': 'R^2, original data',
        'mae_orig': 'MAE, original data'
    }

    def __init__(self, X, y, model, model_name="model", path="./results", 
                 step_hist=100, N=10**4, trial=1, seed=42, 
                 control_strategy="none", control_init=0):
        """
        Creates an instance of the experiment

        :param X: a dataset for regression
        :param y: target variable
        :param model: a class/constructor of the model, should be `callable` which returns and instance of the model with a `fit` method
        :param model_name: a filename to use for figures
        """
        self.X = X
        self.y = y
        self.gbr_tst = []
        self.gbr_base = None
        self.model = model
        self.model_name = model_name
        self.path = path
        self.step_hist = step_hist
        self.N = int(N)
        self.trial = trial
        self.seed = seed
        self.control_strategy = control_strategy
        self.control_add = float(control_init) if control_strategy != "none" else 0

    def prepare_data(self, use_log=False, window_size=0.3, test_size=0.25):
        """
        Initializes the experiment

        :param train_size: size of the sliding window as a portion of the dataset
        :return: None
        """
        self.window_size = float(window_size)
        self.test_size = float(test_size)
        self.use_log = use_log

        self.X_orig, self.X_new, self.y_orig, self.y_new = \
            model_selection.train_test_split(
                self.X,
                np.log(self.y) if self.use_log else self.y,
                train_size=self.window_size)
        
        
        self.X_curr = self.X_orig
        self.y_curr = self.y_orig
        self.mae, self.r2 = [], []
        self.mae_orig, self.r2_orig = [], []
        self.mae_new, self.r2_new = [], []
        self.removed_users_list = []

    def _add_instances(self, X, y, usage=0.9, adherence=0.9):
        """
        This is a generator function (co-routine) for the sliding window loop.
        Works as follows.

        Called once when the loop is initialized.
        Python creates a generator that returns any values provided from this method.
        The method returns the next value via `yield` and continues when `next()` is called on the generator.

        `X` and `y` are set on the first invocation.

        :param X:
        :param y:
        :param usage: how closely users adhere to predictions: `0` means exactly
        :param adherence: share of users to follow prediction
        :return: yields a new sample index from `X`, new price - from `y` or as model predicted
        """

        for sample in np.random.permutation(len(X)):
            if np.random.random() <= float(usage):
                pred = self.gbr.predict([X[sample]])
                new_price = np.random.normal(pred, self.m * float(adherence))[0]
            else:
                new_price = y[sample]

            yield sample, new_price

    def eval_m(self, model, X, y, mae=None, r2=None):
        gbr_pred = model.predict(X)

        mae_v = mean_absolute_error(y, gbr_pred)
        r2_v = r2_score(y, gbr_pred)
        
        if mae is not None:
            mae.append(mae_v)
        if r2 is not None:
            r2.append(r2_v)
        
        return mae_v, r2_v

    def hidden_sampling_experiment(self, adherence=0., usage=1., 
                                   retrain_step=10, test_size=0.25):
        self.X_curr = np.copy(self.X)
        self.y_curr = np.copy(self.y)
        self.y_start = np.copy(self.y)
        #self.X_tr, self.X_tst, self.y_tr, self.y_tst = \
        #    model_selection.train_test_split(self.X_curr, self.y_curr, test_size=test_size,
        #                                     random_state=self.seed)
        self.idx_tst = np.random.choice(range(len(self.X_curr)),
                                        size=int(len(self.X_curr) * test_size),
                                        replace=False)
        self.idx_tr = np.array([idx for idx in range(len(self.X_curr)) if idx not in self.idx_tst])
        self.idx_tr_start = np.copy(self.idx_tst)

        self.gbr_base = self.model()
        self.gbr_base.fit(self.X_curr[self.idx_tr], self.y_curr[self.idx_tr])

        self.gbr = self.model()
        self.gbr.fit(self.X_curr[self.idx_tr], self.y_curr[self.idx_tr])

        self.m, self.r = self.eval_m(self.gbr, self.X_curr[self.idx_tst], 
                                     self.y_curr[self.idx_tst], self.mae, self.r2)
        
        results_file = f"{self.path}/results.csv" 
        with open(results_file, "w") as f:
            f.write(f"t,r2_score,N_t,mean_price\n")
            idx_all = np.hstack([self.idx_tr, self.idx_tst])
            N_t = len(idx_all)
            mean_price = np.mean(self.y_curr[idx_all])
            f.write(f"{0},{self.r:.5f},{N_t},{mean_price}\n")

        for i in tqdm(range(1, self.N+1)):
            idx = np.random.randint(0, len(self.X_curr))
            sample = self.X_curr[idx]
            if np.random.random() <= float(usage):
                pred = self.gbr.predict([sample])
                new_price = np.random.normal(pred, self.m * float(adherence))[0]
                #new_price_control = (1 + self.control_add) * new_price
            else:
                new_price = self.y_curr[idx]
                #new_price_control = new_price

            #if np.random.random() <= np.exp(-0.5*(new_price_control - new_price) / self.y_start[idx]):
            if np.random.random() <= np.exp(-(new_price - self.y_start[idx]) / self.y_start[idx]):
                #self.y_curr[idx] = new_price_control
                self.y_curr[idx] = new_price
                if idx not in np.hstack([self.idx_tr, self.idx_tst]):
                    if idx in self.idx_tr_start:
                        self.idx_tr = np.append(self.idx_tr, idx)
                    else:
                        self.idx_tst = np.append(self.idx_tst, idx)
                self.removed_users_list.append(1)
            else:
                self.idx_tr = np.delete(self.idx_tr, self.idx_tr == idx)
                self.idx_tst = np.delete(self.idx_tst, self.idx_tst == idx)
                self.removed_users_list.append(0)

            if self.control_strategy == "smart_greedy" and np.sum(self.removed_users_list[-5:]) <= 1:
                    self.control_add *= 0.5
            
            if i % int(retrain_step) == 0:
                self.y_curr[self.idx_tr] *= (1 + self.control_add)
                self.y_curr[self.idx_tst] *= (1 + self.control_add)
                X_tr = self.X_curr[self.idx_tr]
                y_tr = self.y_curr[self.idx_tr]
                perm = np.random.permutation(len(y_tr))
                X_tr = X_tr[perm]
                y_tr = y_tr[perm]

                self.gbr = self.model()
                self.gbr.fit(X_tr, y_tr)

                self.m, self.r = self.eval_m(self.gbr, self.X_curr[self.idx_tst],
                                             self.y_curr[self.idx_tst], self.mae, self.r2)
                if self.r < self.r2[-2]:
                    self.control_add *= 0.5

                with open(results_file, "a") as f:
                    idx_all = np.hstack([self.idx_tr, self.idx_tst])
                    N_t = len(idx_all)
                    mean_price = np.mean(self.y_curr[idx_all])
                    f.write(f"{i},{self.r:.5f},{N_t},{mean_price}\n")

                        



class MultipleResults:
    import pandas as pd

    def __init__(self, model_name, **initial_state):
        self.model_name = model_name
        self.state_vars = initial_state
        for k, v in initial_state.items():
            vars(self)[k] = list()

    def add_results(self, **update_state):
        for k in self.state_vars.keys():
            vars(self)[k].extend([{'round':i, k:j} for i, j in enumerate(update_state[k])])

    def plot_multiple_results(self, path):
        for k in self.state_vars.keys():
            data = MultipleResults.pd.DataFrame(data=vars(self)[k])

            plt.figure()
            ax = sb.lineplot(data=data, x="round", y=k)
            ax.set(xlabel='rounds', ylabel=k, title=self.state_vars[k])
            plt.savefig(f"{path}/{self.model_name}-{k}.png")
            data.to_csv(f"{path}/{self.model_name}-{k}.csv")
