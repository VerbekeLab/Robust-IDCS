import numpy as np
import matplotlib.pyplot as plt
from Robust_IDCS.robustIDCS import get_cost_matrix
from sklearn.model_selection import train_test_split
from Toy_Example.generate_data import generate_data
from Toy_Example.metrics import get_metrics
from Toy_Example.train_predict import train_predict_logit, train_predict_cslogit, train_predict_rcslogit

plt.style.use(['science', 'ieee'])

######################
#   OVERVIEW
#   1. Default case
#   1.1 generate data - default
#   1.2 fit logit, cslogit, r-cslogit
#   1.3 metrics
#   1.4 visualize
#
#   2. with outlier
#   2.1 generate data - with outlier
#   2.2 fit logit, cslogit, r-cslogit
#   2.3 metrics
#   2.4 visualize
#
#   3. with noise
#   3.1 generate data - with noise
#   3.2 fit logit, cslogit, r-cslogit
#   3.3 metrics
#   3.4 visualize
######################


def run_toy_example(sample_n, vertical_split, basecost_neg, basecost_pos, cost_neg_x1_coeff, cost_pos_x1_coeff,
                    random_seed, noise_factor, test_size, x1_outlier, x2_outlier, outlier_amount, noise_level):

    """1. default case"""
    '''1.1: Generate data'''
    X,y,A,X_0,X_1,A_0,A_1 = generate_data(sample_n,vertical_split,basecost_neg,basecost_pos,cost_neg_x1_coeff,cost_pos_x1_coeff,random_seed,noise_factor)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    A_train, A_test = train_test_split(A, test_size=test_size, random_state=0)

    """1.2 fit logit, cslogit, r-cslogit"""
    yhat_logit, b_0, w1_0, w2_0 = train_predict_logit(X_train, y_train, X_test)
    yhat_cslogit, b_1, w1_1, w2_1 = train_predict_cslogit(A_train,A_test,X_train,y_train,X_test)
    yhat_rcslogit, b_2, w1_2, w2_2 = train_predict_rcslogit(A_train,A_test,X_train,y_train,X_test)


    """1.3 calculate metrics"""
    costmatrix_test = get_cost_matrix(A_test)
    savings_logit,AUC_logit,F1_logit = get_metrics(yhat_logit,y_test,costmatrix_test)
    print("****** Results Synthetic Data ******")
    print("1. Synthetic data:")
    print('logit savings: \t\t' + str(savings_logit))
    print('logit AUC: \t\t\t' + str(AUC_logit))
    print('logit F1: \t\t\t' + str(F1_logit))

    savings_cslogit,AUC_cslogit,F1_cslogit = get_metrics(yhat_cslogit,y_test,costmatrix_test)
    print('cslogit savings: \t' + str(savings_cslogit))
    print('cslogit AUC: \t\t' + str(AUC_cslogit))
    print('cslogit F1:\t\t\t' + str(F1_cslogit))

    savings_rcslogit,AUC_rcslogit,F1_rcslogit = get_metrics(yhat_rcslogit,y_test,costmatrix_test)
    print('r-cslogit savings:\t' + str(savings_rcslogit))
    print('r-cslogit AUC:\t \t' + str(AUC_rcslogit))
    print('r-cslogit F1: \t\t' + str(F1_rcslogit))

    '''1.4 Visualize'''
    intercept_logit = float(-b_0/w2_0)
    slope_logit = float(-w1_0/w2_0)
    intercept_cslogit = float(-b_1/w2_1)
    slope_cslogit = float(-w1_1/w2_1)
    intercept_rcslogit = float(-b_2/w2_2)
    slope_rcslogit = float(-w1_2/w2_2)

#    X = np.concatenate((X_1, X_0))

    fig, ax = plt.subplots()
    ax.scatter(X_0[:,0], X_0[:,1], color='blue', s=A_0/3, edgecolors=None)
    ax.scatter(X_1[:,0], X_1[:,1], color='red', s=A_1/3, edgecolors=None)

    ax.axhline(y=0, color='grey')
    ax.axvline(x=0, color='grey')
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    fig.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X1')
    plt.ylabel('X2')

    logitmodel_boundary, _ = plt.plot(X, float(slope_logit)*X+float(intercept_logit), '-r', label='logit')
    cslogitmodel_boundary, _ = plt.plot(X, float(slope_cslogit)*X+float(slope_cslogit), '-b', label='cslogit')
    rcslogitmodel_boundary, _ = plt.plot(X, float(slope_rcslogit)*X+float(slope_rcslogit), '-g', label='r-cslogit')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),handles=[logitmodel_boundary, cslogitmodel_boundary,
                                                                   rcslogitmodel_boundary])
    plt.title('Synthetic data default case')
    plt.tight_layout()
    plt.show()

#%%
    """2. Toy example with outlier"""
    '''2.1: Generate data'''
    X, y, A, X_0, X_1, A_0, A_1 = generate_data(sample_n, vertical_split, basecost_neg, basecost_pos, cost_neg_x1_coeff,
                                                cost_pos_x1_coeff, random_seed, noise_factor)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    A_train, A_test = train_test_split(A, test_size=test_size, random_state=0)

    X_train[0, 0] = x1_outlier
    X_train[0, 1] = x2_outlier
    A_train[0] = outlier_amount

    """2.2 fit logit, cslogit, r-cslogit"""
    yhat_logit, b_0, w1_0, w2_0 = train_predict_logit(X_train, y_train, X_test)
    yhat_cslogit, b_1, w1_1, w2_1 = train_predict_cslogit(A_train, A_test, X_train, y_train, X_test)
    yhat_rcslogit, b_2, w1_2, w2_2 = train_predict_rcslogit(A_train, A_test, X_train, y_train, X_test)

    """2.3 Metrics"""
    costmatrix_test = get_cost_matrix(A_test)
    savings_logit,AUC_logit,F1_logit = get_metrics(yhat_logit,y_test,costmatrix_test)
    print("2. Synthetic data with outlier")
    print('logit savings:\t\t' + str(savings_logit))
    print('logit AUC: \t\t\t' + str(AUC_logit))
    print('logit F1: \t\t\t' + str(F1_logit))

    savings_cslogit,AUC_cslogit,F1_cslogit = get_metrics(yhat_cslogit,y_test,costmatrix_test)
    print('cslogit savings:\t' + str(savings_cslogit))
    print('cslogit AUC: \t\t' + str(AUC_cslogit))
    print('cslogit F1:\t \t\t' + str(F1_cslogit))

    savings_rcslogit,AUC_rcslogit,F1_rcslogit = get_metrics(yhat_rcslogit,y_test,costmatrix_test)
    print('r-cslogit savings:\t' + str(savings_rcslogit))
    print('r-cslogit AUC: \t\t' + str(AUC_rcslogit))
    print('r-cslogit F1: \t\t' + str(F1_rcslogit))

    '''2.4 Visualize'''
    intercept_logit = float(-b_0 / w2_0)
    slope_logit = float(-w1_0 / w2_0)
    intercept_cslogit = float(-b_1 / w2_1)
    slope_cslogit = float(-w1_1 / w2_1)
    intercept_rcslogit = float(-b_2 / w2_2)
    slope_rcslogit = float(-w1_2 / w2_2)

    #    X = np.concatenate((X_1, X_0))
    X_1[0, 0] = x1_outlier
    X_1[0, 1] = x2_outlier
    A_1[0] = outlier_amount

    fig, ax = plt.subplots()
    ax.scatter(X_0[:, 0], X_0[:, 1], color='blue', s=A_0/3, edgecolors=None)
    ax.scatter(X_1[:, 0], X_1[:, 1], color='red', s=A_1/3, edgecolors=None)

    ax.axhline(y=0, color='grey')
    ax.axvline(x=0, color='grey')
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    fig.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X1')
    plt.ylabel('X2')

    logitmodel_boundary, _ = plt.plot(X, float(slope_logit) * X + float(intercept_logit), '-r', label='logit')
    cslogitmodel_boundary, _ = plt.plot(X, float(slope_cslogit) * X + float(slope_cslogit), '-b',
                                        label='cslogit')
    rcslogitmodel_boundary, _ = plt.plot(X, float(slope_rcslogit) * X + float(slope_rcslogit), '-g',
                                        label='r-cslogit')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),handles=[logitmodel_boundary, cslogitmodel_boundary, rcslogitmodel_boundary])
    plt.title('Synthetic data with one outlier')
    plt.tight_layout()
    plt.show()


# %%
    """3. Toy example with noise"""
    '''3.1: Generate data'''
    X, y, A, X_0, X_1, A_0, A_1 = generate_data(sample_n, vertical_split, basecost_neg, basecost_pos, cost_neg_x1_coeff,
                                                cost_pos_x1_coeff, random_seed, noise_factor)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    A_train, A_test = train_test_split(A, test_size=test_size, random_state=0)

    mu = 2
    sigma = noise_level
    noise_train = np.random.lognormal(mu,sigma,len(X_train))
    A_train =+ noise_train
    noise_A_0 = np.random.lognormal(mu,sigma,len(A_0))
    A_0 =+ noise_A_0
    noise_A_1 = np.random.lognormal(mu,sigma,len(A_1))
    A_1 =+ noise_A_1

    """3.2 fit logit, cslogit, r-cslogit"""
    yhat_logit, b_0, w1_0, w2_0 = train_predict_logit(X_train, y_train, X_test)
    yhat_cslogit, b_1, w1_1, w2_1 = train_predict_cslogit(A_train, A_test, X_train, y_train, X_test)
    yhat_rcslogit, b_2, w1_2, w2_2 = train_predict_rcslogit(A_train, A_test, X_train, y_train, X_test)

    """3.3 calculate metrics"""
    costmatrix_test = get_cost_matrix(A_test)
    savings_logit,AUC_logit,F1_logit = get_metrics(yhat_logit,y_test,costmatrix_test)
    print("3. Synthetic data with noise:")
    print('logit savings: \t\t' + str(savings_logit))
    print('logit AUC: \t\t\t' + str(AUC_logit))
    print('logit F1: \t\t\t' + str(F1_logit))

    savings_cslogit,AUC_cslogit,F1_cslogit = get_metrics(yhat_cslogit,y_test,costmatrix_test)
    print('cslogit savings: \t' + str(savings_cslogit))
    print('cslogit AUC: \t\t' + str(AUC_cslogit))
    print('cslogit F1: \t\t' + str(F1_cslogit))

    savings_rcslogit,AUC_rcslogit,F1_rcslogit = get_metrics(yhat_rcslogit,y_test,costmatrix_test)
    print('r-cslogit savings: \t' + str(savings_rcslogit))
    print('r-cslogit AUC: \t\t' + str(AUC_rcslogit))
    print('r-cslogit F1: \t\t' + str(F1_rcslogit))

    '''3.4 Visualize'''
    intercept_logit = float(-b_0 / w2_0)
    slope_logit = float(-w1_0 / w2_0)
    intercept_cslogit = float(-b_1 / w2_1)
    slope_cslogit = float(-w1_1 / w2_1)
    intercept_rcslogit = float(-b_2 / w2_2)
    slope_rcslogit = float(-w1_2 / w2_2)

    fig, ax = plt.subplots()
    ax.scatter(X_0[:, 0], X_0[:, 1], color='blue', s=A_0/2, edgecolors=None)
    ax.scatter(X_1[:, 0], X_1[:, 1], color='red', s=A_1/2, edgecolors=None)

    ax.axhline(y=0, color='grey')
    ax.axvline(x=0, color='grey')
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    fig.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X1')
    plt.ylabel('X2')

    logitmodel_boundary, _ = plt.plot(X, float(slope_logit) * X + float(intercept_logit), '-r', label='logit')
    cslogitmodel_boundary, _ = plt.plot(X, float(slope_cslogit) * X + float(slope_cslogit), '-b',
                                        label='cslogit')
    rcslogitmodel_boundary, _ = plt.plot(X, float(slope_rcslogit) * X + float(slope_rcslogit), '-g',
                                         label='r-cslogit')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),handles=[logitmodel_boundary, cslogitmodel_boundary, rcslogitmodel_boundary])
    plt.title('Synthetic data with noise')
    plt.tight_layout()
    plt.show()