import numpy as np
from scipy.stats import sem
from sklearn.linear_model import HuberRegressor
import pandas as pd

def calculate_savings(Predictions ,y_test ,realCost_matrix_test):
    # calculate cost with model
    cost_tn = realCost_matrix_test[:, 0, 0][np.logical_and(Predictions == 0, y_test== 0)].sum()
    cost_fn = realCost_matrix_test[:, 0, 1][np.logical_and(Predictions == 0, y_test == 1)].sum()
    cost_fp = realCost_matrix_test[:, 1, 0][np.logical_and(Predictions == 1, y_test== 0)].sum()
    cost_tp = realCost_matrix_test[:, 1, 1][np.logical_and(Predictions == 1, y_test == 1)].sum()

    cost_with_model = sum((cost_tn, cost_fn, cost_fp, cost_tp))

    # calculate cost without model
    savings_cost_neg = realCost_matrix_test[:, 0, 0][y_test == 0].sum() + realCost_matrix_test[:, 0, 1][y_test == 1].sum()
    savings_cost_pos = realCost_matrix_test[:, 1, 0][y_test == 0].sum() + realCost_matrix_test[:, 1, 1][y_test == 1].sum()

    cost_without_model = min(savings_cost_neg ,savings_cost_pos)

    # calculate savings:
    savings = 1 - (cost_with_model / cost_without_model)

    # return result
    return round(savings,5)

#this function returns a symmetric cost matrix, based on an array of costs
def get_cost_matrix(cost):
    cost_matrix = np.zeros((len(cost), 2, 2))     # cost_matrix [[TN, FN], [FP, TP]]
    cost_matrix[:, 0, 0] = 0
    cost_matrix[:, 0, 1] = cost
    cost_matrix[:, 1, 0] = cost
    cost_matrix[:, 1, 1] = 0

    return cost_matrix

#Function to make predictions of Amounts from df : df(X,y,Amount) -> df(X,y,Amount,Amount_pred)
def enhance_df(df):

    model_0 = HuberRegressor(max_iter=1000)
    model_1 = HuberRegressor(max_iter=1000)

    df_0 = df[df.y == 0]
    model_0.fit(df_0.loc[:, df_0.columns != 'Amount'], df_0['Amount'])
    Ahat_0_array = model_0.predict(df_0.loc[:, df_0.columns != 'Amount'])
    Ahat_0_array[Ahat_0_array<0] = 0

    df_1 = df[df.y == 1]
    model_1.fit(df_1.loc[:, df_1.columns != 'Amount'], df_1['Amount'])
    Ahat_1_array = model_1.predict(df_1.loc[:, df_1.columns != 'Amount'])
    Ahat_1_array[Ahat_1_array < 0] = 0

    df_0['Amount_pred'] = Ahat_0_array
    df_1['Amount_pred'] = Ahat_1_array
    prediction = pd.concat([df_0,df_1])
    df['Amount_pred'] = prediction['Amount_pred']

    return df

#Function to detect and impute conditionally outlying amount
def impute_amounts(df):

    std = sem(df['Amount'])
    df['resid'] = df['Amount'] - df['Amount_pred']
    df['resid_std'] = df['resid']/std
    df['resid_std_abs'] = abs(df['resid_std'])
    df.loc[df.resid_std_abs > 3, 'Amount'] = df['Amount_pred']
    print("")
    df = df.drop(columns=['Amount_pred','resid','resid_std','resid_std_abs'])
    return df
