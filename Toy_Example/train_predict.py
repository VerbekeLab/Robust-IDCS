import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from Robust_IDCS.robustIDCS import enhance_df, impute_amounts
from methodologies.cs_logit import CSLogit

def get_init_theta(X_train, y_train):
    logitModel = LogisticRegression(max_iter=1000, verbose=0)
    logitModel.fit(X_train, y_train)
    init_theta = np.insert(logitModel.coef_, 0, values=logitModel.intercept_)

    return init_theta

#this function returns a symmetric cost matrix, based on an array of costs
def get_cost_matrix(cost):
    cost_matrix = np.zeros((len(cost), 2, 2))     # cost_matrix [[TN, FN], [FP, TP]]
    cost_matrix[:, 0, 0] = 0
    cost_matrix[:, 0, 1] = cost
    cost_matrix[:, 1, 0] = cost
    cost_matrix[:, 1, 1] = 0

    return cost_matrix

def train_predict_logit(X_train, y_train, X_test):
    logitModel = LogisticRegression(max_iter=100, verbose=0)
    logitModel.fit(X_train, y_train)
    logit_proba = logitModel.predict_proba(X_test)
    logit_proba = np.delete(logit_proba, [0], axis=1)  # only keep the probabilities for y==1, not both y==0 and y==1
    logit_proba = logit_proba.flatten()
    logit_binary = logitModel.predict(X_test)
    b = logitModel.intercept_[0]
    w1, w2 = logitModel.coef_.T

    return logit_binary, b, w1, w2

def train_predict_cslogit(cost_train,cost_test,X_train,y_train,X_test):
    init_theta = get_init_theta(X_train,y_train)
    cost_matrix_train = get_cost_matrix(cost_train)
    CSlogitModel = CSLogit(initial_theta=init_theta, obj='aec')
    CSlogitModel.fitting(x=X_train, y=y_train, cost_matrix=cost_matrix_train)
    CSlogit_proba = CSlogitModel.predict(X_test)
    CSlogit_binary = CSlogit_proba.round()
    b = CSlogitModel.theta_opt[0]
    w1 = CSlogitModel.theta_opt[1]
    w2 = CSlogitModel.theta_opt[2]
    return CSlogit_binary, b, w1, w2

def train_predict_rcslogit(cost_train,cost_test,X_train,y_train,X_test):
    init_theta = get_init_theta(X_train,y_train)
    cost_matrix_train = get_cost_matrix(cost_train)
    # create df: a merge of x, y and amount
    df = pd.DataFrame(data=X_train)
    df['y'] = y_train
    df['Amount'] = cost_train
    df_enhanced = enhance_df(df)
    # step 2: replace outliers with estimated amounts

    df = impute_amounts(df_enhanced)
    # step 3: return cost_matrix_transformed
    cost_matrix_transformed = get_cost_matrix(df['Amount'])
    rCSlogitModel = CSLogit(initial_theta=init_theta, obj='aec')
    rCSlogitModel.fitting(x=X_train, y=y_train, cost_matrix=cost_matrix_transformed)
    # predict with cslogit
    rCSlogit_proba = rCSlogitModel.predict(X_test)
    rCSlogit_binary = rCSlogit_proba.round()
    #retrieve model parameters
    b = rCSlogitModel.theta_opt[0]
    w1 = rCSlogitModel.theta_opt[1]
    w2 = rCSlogitModel.theta_opt[2]

    return rCSlogit_binary, b, w1, w2