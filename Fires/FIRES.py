from sklearn.naive_bayes import GaussianNB
from skmultiflow.lazy import KNNClassifier
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.data import FileStream
from skmultiflow.neural_networks import PerceptronMask
from sklearn.metrics import accuracy_score

import stability as st
import numpy as np
from warnings import warn
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from scipy.io import arff
import warnings


# import Stability as st
class FIRES:
    def __init__(self, n_total_ftr, target_values, mu_init=0, sigma_init=1, penalty_s=0.01, penalty_r=0.01, epochs=1,
                 lr_mu=0.01, lr_sigma=0.01, scale_weights=True, model='probit'):
        """
        FIRES: Fast, Interpretable and Robust Evaluation and Selection of features
        cite:
        Haug et al. 2020. Leveraging Model Inherent Variable Importance for Stable Online Feature Selection.
        In Proceedings of the 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’20),
        August 23–27, 2020, Virtual Event, CA, USA.
        :param n_total_ftr: (int) Total no. of features
        :param target_values: (np.ndarray) Unique target values (class labels)
        :param mu_init: (int/np.ndarray) Initial importance parameter
        :param sigma_init: (int/np.ndarray) Initial uncertainty parameter
        :param penalty_s: (float) Penalty factor for the uncertainty (corresponds to gamma_s in the paper)
        :param penalty_r: (float) Penalty factor for the regularization (corresponds to gamma_r in the paper)
        :param epochs: (int) No. of epochs that we use each batch of observations to update the parameters
        :param lr_mu: (float) Learning rate for the gradient update of the importance
        :param lr_sigma: (float) Learning rate for the gradient update of the uncertainty
        :param scale_weights: (bool) If True, scale feature weights into the range [0,1]
        :param model: (str) Name of the base model to compute the likelihood (default is 'probit')
        """

        self.n_total_ftr = n_total_ftr
        self.target_values = target_values
        self.mu = np.ones(n_total_ftr) * mu_init
        self.sigma = np.ones(n_total_ftr) * sigma_init
        self.penalty_s = penalty_s
        self.penalty_r = penalty_r
        self.epochs = epochs
        self.lr_mu = lr_mu
        self.lr_sigma = lr_sigma
        self.scale_weights = scale_weights
        self.model = model

        # Additional model-specific parameters
        self.model_param = {}

        # Probit model
        if self.model == 'probit' and tuple(target_values) != (-1, 1):
            if len(np.unique(target_values)) == 2:
                self.model_param['probit'] = True  # Indicates that we need to encode the target variable into {-1,1}
                warn('FIRES WARNING: The target variable will be encoded as: {} = -1, {} = 1'.format(
                    self.target_values[0], self.target_values[1]))
            else:
                raise ValueError('The target variable y must be binary.')

        # ### ADD YOUR OWN MODEL PARAMETERS HERE #######################################
        # if self.model == 'your_model':
        #    self.model_param['your_model'] = {}
        ################################################################################

    def weigh_features(self, x, y):
        """
        Compute feature weights, given a batch of observations and corresponding labels
        :param x: (np.ndarray) Batch of observations
        :param y: (np.ndarray) Batch of labels
        :return: feature weights
        :rtype np.ndarray
        """

        # Update estimates of mu and sigma given the predictive model
        if self.model == 'probit':
            self.__probit(x, y)
        # ### ADD YOUR OWN MODEL HERE ##################################################
        # elif self.model == 'your_model':
        #    self.__yourModel(x, y)
        ################################################################################
        else:
            raise NotImplementedError('The given model name does not exist')

        # Limit sigma to range [0, inf]
        if sum(n < 0 for n in self.sigma) > 0:
            self.sigma[self.sigma < 0] = 0
            warn('Sigma has automatically been rescaled to [0, inf], because it contained negative values.')

        # Compute feature weights
        return self.__compute_weights()

    def __probit(self, x, y):
        """
        Update the distribution parameters mu and sigma by optimizing them in terms of the (log) likelihood.
        Here we assume a Bernoulli distributed target variable. We use a Probit model as our base model.
        This corresponds to the FIRES-GLM model in the paper.
        :param x: (np.ndarray) Batch of observations (numeric values only, consider normalizing data for better results)
        :param y: (np.ndarray) Batch of labels: type binary, i.e. {-1,1} (bool, int or str will be encoded accordingly)
        """

        for epoch in range(self.epochs):
            # Shuffle the observations
            random_idx = np.random.permutation(len(y))
            x = x[random_idx]
            y = y[random_idx]

            # Encode target as {-1,1}
            if 'probit' in self.model_param:
                y[y == self.target_values[0]] = -1
                y[y == self.target_values[1]] = 1

            # Iterative update of mu and sigma
            try:
                # Helper functions
                dot_mu_x = np.dot(x, self.mu)
                rho = np.sqrt(1 + np.dot(x ** 2, self.sigma ** 2))

                # Gradients
                nabla_mu = norm.pdf(y / rho * dot_mu_x) * (y / rho * x.T)
                nabla_sigma = norm.pdf(y / rho * dot_mu_x) * (
                            - y / (2 * rho ** 3) * 2 * (x ** 2 * self.sigma).T * dot_mu_x)

                # Marginal Likelihood
                marginal = norm.cdf(y / rho * dot_mu_x)

                # Update parameters
                self.mu += self.lr_mu * np.mean(nabla_mu / marginal, axis=1)
                self.sigma += self.lr_sigma * np.mean(nabla_sigma / marginal, axis=1)
            except TypeError as e:
                raise TypeError('All features must be a numeric data type.') from e

    '''
    # ### ADD YOUR OWN MODEL HERE ##################################################
    def __yourModel(self):
        """ 
        Your own model description.

        :param x: (np.ndarray) Batch of observations
        :param y: (np.ndarray) Batch of labels
        """

        gradientMu = yourFunction()  # Gradient of the (log) likelihood with respect to mu
        gradientSigma = yourFunction()  # Gradient of the (log) likelihood with respect to sigma
        self.mu += self.lr_mu * gradientMu
        self.sigma += self.lr_sigma * gradientSigma
    ################################################################################
    '''

    def __compute_weights(self):
        """
        Compute optimal weights according to the objective function proposed in the paper.
        We compute feature weights in a trade-off between feature importance and uncertainty.
        Thereby, we aim to maximize both the discriminative power and the stability/robustness of feature weights.
        :return: feature weights
        :rtype np.ndarray
        """

        # Compute optimal weights
        weights = (self.mu ** 2 - self.penalty_s * self.sigma ** 2) / (2 * self.penalty_r)

        if self.scale_weights:  # Scale weights to [0,1]
            weights = MinMaxScaler().fit_transform(weights.reshape(-1, 1)).flatten()

        return weights


warnings.simplefilter(action='ignore', category=FutureWarning)

batch_sizes = [25, 50, 75, 100]
fractions = [0.1, 0.15, 0.2]

classifiers = {
        "NEURAL-NETWORKS": PerceptronMask(),
        "KNN": KNNClassifier(n_neighbors=10),
        "HT": HoeffdingTreeClassifier(),
        "NB": GaussianNB()
    }


def apply_fires(classifier_name, df_name, tgt_index, epochs=1):
    final_stab_lst = []
    final_acc_lst = []

    for batch_size in batch_sizes:
        for frac_selected_ftr in fractions:
            # Load data as scikit-multiflow FileStream
            # NOTE: FIRES accepts only numeric values. Please one-hot-encode or factorize string/char variables
            # Additionally, we suggest users to normalize all features, e.g. by using scikit-learn's MinMaxScaler()
            stream = FileStream(df_name, target_idx=tgt_index)
            stream.prepare_for_use()

            # Initial fit of the predictive model
            predictor = classifiers[classifier_name]
            x, y = stream.next_sample(batch_size=batch_size)
            predictor.partial_fit(x, y, stream.target_values)

            # Initialize FIRES
            fires_model = FIRES(n_total_ftr=stream.n_features,  # Total no. of features
                                target_values=stream.target_values,  # Unique target values (class labels)
                                mu_init=0,  # Initial importance parameter
                                sigma_init=1,  # Initial uncertainty parameter
                                penalty_s=0.01,
                                # Penalty factor for the uncertainty (corresponds to gamma_s in the paper)
                                penalty_r=0.01,
                                # Penalty factor for the regularization (corresponds to gamma_r in the paper)
                                epochs=epochs,
                                # No. of epochs that we use each batch of observations to update the parameters
                                lr_mu=0.01,  # Learning rate for the gradient update of the importance
                                lr_sigma=0.01,  # Learning rate for the gradient update of the uncertainty
                                scale_weights=True,  # If True, scale feature weights into the range [0,1]
                                model='probit')  # Name of the base model to compute the likelihood

            # Variables for calculating the average accuracy and stability per time step
            n_selected_ftr = round(frac_selected_ftr * stream.n_features)
            sum_acc, sum_stab, count_time_steps, stability_mat = 0, 0, 0, []
            stability_counter = 0
            start_window, end_window = 0, 9

            while stream.has_more_samples():
                # Load a new sample
                x, y = stream.next_sample(batch_size=batch_size)

                # Select features
                ftr_weights = fires_model.weigh_features(x, y)  # Get feature weights with FIRES
                ftr_selection = np.argsort(ftr_weights)[::-1][:n_selected_ftr]

                # Truncate x (retain only selected features, 'remove' all others, e.g. by replacing them with 0)
                x_reduced = np.zeros(x.shape)
                x_reduced[:, ftr_selection] = x[:, ftr_selection]

                # Prepare x to stability
                x_binary = np.zeros(stream.n_features)
                x_binary[ftr_selection] = 1
                stability_mat.append(x_binary)

                # Test
                y_pred = predictor.predict(x_reduced)
                acc_score = accuracy_score(y, y_pred)
                # print(acc_score)

                # Sum all the accuracy scores
                sum_acc = sum_acc + acc_score

                # Sum all the stabilty scores (shifting window = 10)
                if len(stability_mat) >= 10:
                    sum_stab = sum_stab + st.getStability(stability_mat[start_window:end_window])
                    # print(st.getStability(stability_mat[start_window:end_window]))
                    start_window += 1
                    end_window += 1
                    stability_counter += 1

                # Sum the time steps
                count_time_steps += 1

                # Train
                predictor.partial_fit(x_reduced, y)

            # Average accuracy  and stability
            avg_acc = sum_acc / count_time_steps
            avg_stab = sum_stab / (stability_counter)
            print(f'avg acc score: {avg_acc}')
            print(f'stability score: {avg_stab}')

            final_stab_lst.append(avg_stab)
            final_acc_lst.append(avg_acc)
            # Restart the FileStream
            stream.restart()

    print(f'Final avg acc score: {sum(final_acc_lst) / len(final_acc_lst)}')
    print(f'Final avg stab score: {sum(final_stab_lst) / len(final_stab_lst)}')

if __name__ == "__main__":
    apply_fires("NEURAL-NETWORKS", df_name='/Users/samuelbenichou/Downloads/normalize 2/electricity_data.csv', tgt_index=0, epochs=1)