from sklearn.metrics import roc_curve, auc, f1_score
from Robust_IDCS.robustIDCS import calculate_savings

def get_metrics(y_hat,y_test,costmatrix_test):
    savings = calculate_savings(y_hat, y_test, costmatrix_test)
    fpr, tpr, _ = roc_curve(y_test, y_hat)
    AUC = auc(fpr, tpr)
    F1 = f1_score(y_test,y_hat)

    savings = round(savings,5)
    AUC = round(AUC,5)
    F1 = round(F1,5)
    return savings, AUC, F1