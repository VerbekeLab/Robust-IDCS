import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
import time

#Synthetic data:
from Toy_Example.toy_example import run_toy_example

#Framework:
from preprocessing.preprocessing import convert_categorical_variables, standardize, handle_missing_data, preprocess_credit_card_data
from experiments.experimental_design import experimental_design
from performance_metrics.performance_metrics import get_performance_metrics, evaluate_experiments, cost_with_algorithm

# Models:
from methodologies.cs_logit import CSLogit
from sklearn.linear_model import LogisticRegression

#Warnings:
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

class Experiment:
    def __init__(self, settings, datasets, methodologies, toyexample, evaluators):
        self.settings = settings

        self.l1 = self.settings['l1_regularization']
        self.lambda1_list = self.settings['lambda1_options']
        self.l2 = self.settings['l2_regularization']
        self.lambda2_list = self.settings['lambda2_options']
        self.neurons_list = self.settings['neurons_options']

        if self.l1 and self.l2:
            raise ValueError('Only l1 or l2 regularization allowed, not both!')

        self.datasets = datasets
        self.methodologies = methodologies
        self.toyexample = toyexample

        self.evaluators = evaluators

        self.results_tr_instance = {}
        self.results_tr_instance_calibrated = {}
        self.results_tr_class = {}
        self.results_tr_class_calibrated = {}
        self.results_tr_class_imb = {}
        self.results_tr_empirical_id = {}
        self.results_tr_empirical_cd = {}
        self.results_tr_empirical_f1 = {}
        self.results_tr_insensitive = {}

    @ignore_warnings(category=ConvergenceWarning)
    def run(self, directory):
        """
        DISPLAY TOY EXAMPLE
        """
        if self.toyexample['toy_example']:
            sample_n = self.toyexample['sample_n']
            vertical_split = self.toyexample['vertical_split']
            basecost_neg = self.toyexample['basecost_neg']
            basecost_pos = self.toyexample['basecost_pos']
            cost_neg_x1_coeff = self.toyexample['cost_neg_x1_coeff']
            cost_pos_x1_coeff = self.toyexample['cost_pos_x1_coeff']
            random_seed = self.toyexample['random_seed']
            noise_factor = self.toyexample['noise_factor']
            test_size = self.toyexample['test_size']
            x1_outlier = self.toyexample['x1_outlier']
            x2_outlier = self.toyexample['x2_outlier']
            outlier_amount = self.toyexample['outlier_amount']
            noise_level = self.toyexample['noise_level']
            run_toy_example(sample_n, vertical_split, basecost_neg, basecost_pos, cost_neg_x1_coeff, cost_pos_x1_coeff,
                            random_seed, noise_factor, test_size, x1_outlier, x2_outlier, outlier_amount, noise_level)

        """
        LOAD AND PREPROCESS DATA
        """

        print('\n****** Experiments Real Data ******')
        print('Loading Kaggle creditcard dataset')

        if self.datasets['kaggle credit card fraud']:
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_credit_card_data(fixed_cost=10)

        else:
            raise Exception('No dataset specified')

        """
        RUN EXPERIMENTS
        """

        # Prepare the cross-validation procedure
        folds = self.settings['folds']
        repeats = self.settings['repeats']
        print('Building classification model ('+str(folds)+' folds, '+str(repeats)+' repeats)' )
        rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=0)
        prepr = experimental_design(labels, amounts)

        # Prepare the evaluation matrices
        n_methodologies = sum(self.methodologies.values())
        for key in self.evaluators.keys():
            if self.evaluators[key]:
                self.results_tr_instance[key] = np.empty(shape=(n_methodologies, folds * repeats), dtype='object')
                self.results_tr_instance_calibrated[key] = np.empty(shape=(n_methodologies, folds * repeats),
                                                                    dtype='object')
                self.results_tr_class[key] = np.empty(shape=(n_methodologies, folds * repeats), dtype='object')
                self.results_tr_class_calibrated[key] = np.empty(shape=(n_methodologies, folds * repeats),
                                                                 dtype='object')
                self.results_tr_class_imb[key] = np.empty(shape=(n_methodologies, folds * repeats), dtype='object')
                self.results_tr_empirical_id[key] = np.empty(shape=(n_methodologies, folds * repeats), dtype='object')
                self.results_tr_empirical_cd[key] = np.empty(shape=(n_methodologies, folds * repeats), dtype='object')
                self.results_tr_empirical_f1[key] = np.empty(shape=(n_methodologies, folds * repeats), dtype='object')
                self.results_tr_insensitive[key] = np.empty(shape=(n_methodologies, folds * repeats), dtype='object')

        for i, (train_val_index, test_index) in enumerate(rskf.split(covariates, prepr)):
            print('----------------------------------------------------------------')
            print('Cross validation: ' + str(i + 1))

            index = 0

            x_train_val, x_test = covariates.iloc[train_val_index], covariates.iloc[test_index]
            y_train_val, y_test = labels[train_val_index], labels[test_index]
            amounts_train_val, amounts_test = amounts[train_val_index], amounts[test_index]
            cost_matrix_train_val, cost_matrix_test = cost_matrix[train_val_index, :], cost_matrix[test_index, :]

            # Split training and validation set (based on instance-dependent costs)
            train_ratio = 1 - self.settings['val_ratio']
            skf = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=42)
            prepr_val = experimental_design(y_train_val, amounts_train_val)

            for train_index, val_index in skf.split(x_train_val, prepr_val):
                x_train, x_val = x_train_val.iloc[train_index], x_train_val.iloc[val_index]
                y_train, y_val = y_train_val[train_index], y_train_val[val_index]
                cost_matrix_train, cost_matrix_val = cost_matrix_train_val[train_index, :], cost_matrix_train_val[
                                                                                            val_index, :]

            x_train, x_val, x_test, categorical_variables = handle_missing_data(x_train, x_val, x_test,
                                                                                categorical_variables)
            x_train, x_val, x_test = convert_categorical_variables(x_train, y_train, x_val, x_test,
                                                                   categorical_variables)
            x_train, x_val, x_test = standardize(x_train=x_train, x_val=x_val, x_test=x_test)

            #Add outlier to dataset
            outlier_index = self.datasets['outlier_index']
            outlier_amount = self.datasets['outlier_size']

            #Change label of outlier
            y_train[outlier_index] = 1 - y_train[outlier_index]
            cost_matrix_train[outlier_index, 0, 0] = 0
            cost_matrix_train[outlier_index, 0, 1] = outlier_amount
            cost_matrix_train[outlier_index, 1, 0] = outlier_amount
            cost_matrix_train[outlier_index, 1, 1] = 0

            threshold_cost_ins = np.repeat(0.5, len(y_test))

            # Define evaluation procedure for threshold = 0.5
            def evaluate_model(proba_val, proba, j, index, info):

                # Cost-insensitive threshold:
                pred = (proba > threshold_cost_ins).astype(int)
                self.results_tr_insensitive = get_performance_metrics(self.evaluators, self.results_tr_insensitive, j,
                                                                      index, cost_matrix_test, y_test, proba, pred,
                                                                      info)

            # Logistic regression
            if self.methodologies['logit']:
                print('\tlogistic regression')
                # Get initial estimate for theta and create model
                init_logit = LogisticRegression(penalty='none', max_iter=5000, verbose=0, solver='sag', n_jobs=-1)
                init_logit.fit(x_train, y_train)
                init_theta = np.insert(init_logit.coef_, 0, values=init_logit.intercept_)

                logit = CSLogit(init_theta, obj='ce')

                # Tune regularization parameters, if necessary
                logit.tune(self.l1, self.lambda1_list, self.l2, self.lambda2_list, x_train, y_train, cost_matrix_train,
                           x_val, y_val, cost_matrix_val)

                lambda1 = logit.lambda1
                lambda2 = logit.lambda2

                start = time.perf_counter()
                logit.fitting(x_train, y_train, cost_matrix_train)

                end = time.perf_counter()

                logit_proba = logit.predict(x_test)
                logit_proba_val = logit.predict(x_val)

                info = {'time': end - start, 'lambda1': lambda1, 'lambda2': lambda2, 'n_neurons': 0}

                evaluate_model(logit_proba_val, logit_proba, i, index, info)

                index += 1

            # Cost-sensitive logistic regression
            if self.methodologies['cslogit']:
                print('\tcslogit')
                try:
                    init_logit
                except NameError:
                    init_logit = LogisticRegression(penalty='none', max_iter=5000, verbose=False, solver='sag',
                                                    n_jobs=-1)
                    init_logit.fit(x_train, y_train)
                    init_theta = np.insert(init_logit.coef_, 0, values=init_logit.intercept_)

                cslogit = CSLogit(init_theta, obj='aec')

                cslogit.tune(self.l1, self.lambda1_list, self.l2, self.lambda2_list, x_train, y_train,
                                    cost_matrix_train, x_val, y_val, cost_matrix_val)

                lambda1 = cslogit.lambda1
                lambda2 = cslogit.lambda2

                start = time.perf_counter()
                cslogit.fitting(x_train, y_train, cost_matrix_train)

                end = time.perf_counter()

                cslogit_proba = cslogit.predict(x_test)
                cslogit_proba_val = cslogit.predict(x_val)

                info = {'time': end - start, 'lambda1': lambda1, 'lambda2': lambda2, 'n_neurons': 0}

                evaluate_model(cslogit_proba_val, cslogit_proba, i, index, info)

                index += 1

            # Robust cost-sensitive logistic regression
            if self.methodologies['cslogit_robust']:
                print('\trobust cslogit')
                try:
                    init_logit
                except NameError:
                    init_logit = LogisticRegression(penalty='none', max_iter=5000, verbose=False, solver='sag',
                                                    n_jobs=-1)
                    init_logit.fit(x_train, y_train)
                    init_theta = np.insert(init_logit.coef_, 0, values=init_logit.intercept_)

                cslogit = CSLogit(init_theta, obj='aec', robust=True)

                cslogit.tune(self.l1, self.lambda1_list, self.l2, self.lambda2_list, x_train, y_train,
                                    cost_matrix_train, x_val, y_val, cost_matrix_val)

                lambda1 = cslogit.lambda1
                lambda2 = cslogit.lambda2

                start = time.perf_counter()
                #Documentation: the .fitting() method contains extra rule if robust method
                cslogit.fitting(x_train, y_train, cost_matrix_train)

                end = time.perf_counter()

                cslogit_proba = cslogit.predict(x_test)
                cslogit_proba_val = cslogit.predict(x_val)

                info = {'time': end - start, 'lambda1': lambda1, 'lambda2': lambda2, 'n_neurons': 0}

                evaluate_model(cslogit_proba_val, cslogit_proba, i, index, info)

                index += 1

    def evaluate(self, directory):
        """
        EVALUATION
        """
        print('\n------ Evaluation of classifiers ------')


        with open(str(directory + 'summary.txt'), 'a') as file:
            file.write('\n*** Cost-insensitive thresholds ***\n')
        print('\n -- threshold = 0.5 -- \n')
        evaluate_experiments(evaluators=self.evaluators,
                             methodologies=self.methodologies,
                             evaluation_matrices=self.results_tr_insensitive,
                             directory=directory, name='ins')

