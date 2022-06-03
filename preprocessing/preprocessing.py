import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import woe
import time

# Set random seed
random.seed(2)
# Suppress pd SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'


def convert_categorical_variables(x_train, y_train, x_val, x_test, categorical_variables):

    woe_encoder = woe.WOEEncoder(verbose=1, cols=categorical_variables)
    woe_encoder.fit(x_train, y_train)
    x_train = woe_encoder.transform(x_train)
    x_val = woe_encoder.transform(x_val)
    x_test = woe_encoder.transform(x_test)

    return x_train, x_val, x_test


def standardize(x_train, x_val, x_test):
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train_scaled = scaler.transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    # For compatibility with xgboost package: make contiguous arrays
    x_train_scaled = np.ascontiguousarray(x_train_scaled)
    x_val_scaled = np.ascontiguousarray(x_val_scaled)
    x_test_scaled = np.ascontiguousarray(x_test_scaled)

    return x_train_scaled, x_val_scaled, x_test_scaled


def handle_missing_data(df_train, df_val, df_test, categorical_variables):

    for key in df_train.keys():
        # If variable has > 90% missing values: delete
        if df_train[key].isna().mean() > 0.9:
            df_train = df_train.drop(key, 1)
            df_val = df_val.drop(key, 1)
            df_test = df_test.drop(key, 1)

            if key in categorical_variables:
                categorical_variables.remove(key)
            continue

        # Handle other missing data:
        #   Categorical variables: additional category '-1'
        if key in categorical_variables:
            df_train[key] = df_train[key].fillna('-1')
            df_val[key] = df_val[key].fillna('-1')
            df_test[key] = df_test[key].fillna('-1')
        #   Continuous variables: median imputation
        else:
            median = df_train[key].median()
            df_train[key] = df_train[key].fillna(median)
            df_val[key] = df_val[key].fillna(median)
            df_test[key] = df_test[key].fillna(median)

    assert df_train.isna().sum().sum() == 0 and df_val.isna().sum().sum() == 0 and df_test.isna().sum().sum() == 0

    return df_train, df_val, df_test, categorical_variables

def preprocess_credit_card_data(fixed_cost, eda=False):
    """
    Load the Kaggle credit card dataset
    """
    try:
        creditcard = pd.read_csv('data/Kaggle Credit Card Fraud/creditcard.csv')
    except FileNotFoundError:
        creditcard = pd.read_csv('../data/Kaggle Credit Card Fraud/creditcard.csv')

    # Preprocessing
    creditcard = creditcard[creditcard['Amount'] != 0]  # remove transactions with zero Amount
    creditcard = creditcard.drop('Time', 1)  # remove variable Time

    amounts = creditcard['Amount'].values
    labels = creditcard['Class'].values

    cols = list(creditcard.columns)  # rearrange some columns - amount final column
    a, b = cols.index('Amount'), cols.index('Class')
    cols[b], cols[a] = cols[a], cols[b]
    creditcard = creditcard[cols]

    covariates = creditcard.drop(['Class'], axis=1)

    scaler = StandardScaler()
    covariates_scaled = scaler.fit_transform(covariates)

    cost_matrix = np.zeros((len(covariates), 2, 2))     # cost_matrix [[TN, FN], [FP, TP]]
    cost_matrix[:, 0, 0] = 0
    cost_matrix[:, 0, 1] = amounts
    cost_matrix[:, 1, 0] = amounts
    cost_matrix[:, 1, 1] = 0

    return covariates, labels, amounts, cost_matrix, []  # No categorical variables
