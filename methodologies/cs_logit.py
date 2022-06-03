import numpy as np
import pandas as pd
import scipy.optimize
from scipy.special import expit, xlog1py
from Robust_IDCS.robustIDCS import enhance_df, impute_amounts

#this function returns a symmetric cost matrix, based on an array of costs
def get_cost_matrix(cost):
    cost_matrix = np.zeros((len(cost), 2, 2))     # cost_matrix [[TN, FN], [FP, TP]]
    cost_matrix[:, 0, 0] = 0
    cost_matrix[:, 0, 1] = cost
    cost_matrix[:, 1, 0] = cost
    cost_matrix[:, 1, 1] = 0

    return cost_matrix

class CSLogit:

    def __init__(self, initial_theta, lambda1=0, lambda2=0, theta_opt=None, obj='aec', robust=False):

        self.initial_theta = initial_theta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.theta_opt = theta_opt
        self.obj = obj
        self.robust = robust

    def transform_cost_matrix(self, x, y, cost_matrix):
        # Transform cost matrix
        # create df: a merge of x, y and amount
        amounts = cost_matrix[:, 0, 1]
        df = pd.DataFrame(data=x)
        df['y'] = y
        df['Amount'] = amounts

        #step 1: fit model on df, return df_enhanced
        df_enhanced = enhance_df(df)
        #step 2: replace conditionally outlying amounts with estimated amounts
        df = impute_amounts(df_enhanced)
        #step 3: return cost_matrix_transformed
        cost_matrix_transformed = get_cost_matrix(df['Amount'])
        return cost_matrix_transformed

    def fitting(self, x, y, cost_matrix):
        #In the case of robust version, transform the cost_matrix first
        if self.robust:
            cost_matrix = self.transform_cost_matrix(x, y, cost_matrix)

        if self.obj == 'weightedce':
            misclass_cost = np.zeros(len(y))
            misclass_cost[y == 0] = cost_matrix[:, 1, 0][y == 0]
            misclass_cost[y == 1] = cost_matrix[:, 0, 1][y == 1]

        def get_obj_func(theta):
            if self.obj == 'ce':
                return self.ce(theta, x, y)
            if self.obj == 'ce10':
                return self.ce10(theta, x, y)
            elif self.obj == 'weightedce':
                return self.weighted_ce(theta, x, y, misclass_cost)
            elif self.obj == 'aec':
                return self.aec(theta, x, y, cost_matrix)
            elif self.obj == 'logaec':
                return self.logaec(theta, x, y, cost_matrix)
            else:
                raise Exception('No correct loss specified')

        self.theta, func_min, _ = self.optimization(get_obj_func, self.initial_theta)

        return func_min

    def optimization(self, obj_func, initial_theta):
        opt_res = scipy.optimize.minimize(obj_func, initial_theta, method='L-BFGS-B',
                                          options={'ftol': 1e-6, 'disp': False})
        theta_opt, func_min, n_iter = opt_res.x, opt_res.fun, opt_res.nfev

        self.theta_opt = theta_opt

        return theta_opt, func_min, n_iter

    def ce(self, theta, x, y_true):
        scores = expit(theta[0] + x.dot(theta[1:]))

        eps = 1e-9  # small value to avoid log(0)

        ce = - (y_true * np.log(scores + eps) + (1 - y_true) * np.log(1 - scores + eps))

        # Add regularization
        ce += self.lambda1 * np.sum(abs(theta[1:])) + self.lambda2 * np.sum(theta[1:] ** 2)

        return ce.mean()

    def ce10(self, theta, x, y_true):
        scores = expit(theta[0] + x.dot(theta[1:]))

        eps = 1e-9  # small value to avoid log(0)

        ce = - (y_true * np.log10(scores + eps) + (1 - y_true) * np.log10(1 - scores + eps))

        # Add regularization
        ce += self.lambda1 * np.sum(abs(theta[1:])) + self.lambda2 * np.sum(theta[1:] ** 2)

        return ce.mean()

    def weighted_ce(self, theta, x, y_true, cost_misclass):
        # Weighted CE: Misclassification costs as weights
        scores = expit(theta[0] + x.dot(theta[1:]))

        eps = 1e-9  # small value to avoid log(0)

        # Normalized cost of misclassification per instance
        ce = - (y_true * np.log(scores + eps) + (1 - y_true) * np.log(1 - scores + eps))
        weightedce = cost_misclass * ce

        # Add regularization
        weightedce += self.lambda1 * np.sum(abs(theta[1:])) + self.lambda2 * np.sum(theta[1:] ** 2)

        return weightedce.mean()

    def aec(self, theta, x, y, cost_matrix):  # Average Expected Cost

        scores = expit(theta[0] + x.dot(theta[1:]))  # eq 9

        # objective value = average expected cost + lambd * L1 norm of betas (without intercept)
        # eq 7 and 13
        aec = y * (scores * cost_matrix[:, 1, 1] + (1 - scores) * cost_matrix[:, 0, 1]) \
            + (1 - y) * (scores * cost_matrix[:, 1, 0] + (1 - scores) * cost_matrix[:, 0, 0])

        # Add regularization
        aec += self.lambda1 * np.sum(abs(theta[1:])) + self.lambda2 * np.sum(theta[1:] ** 2)

        # print(aec.mean())

        return aec.mean()

    def logaec(self, theta, x, y, cost_matrix):  # Average Expected Cost

        scores = expit(theta[0] + x.dot(theta[1:]))  # eq 9

        eps = 1e-9

        # objective value = logaec + lambd * L1 norm of betas (without intercept)
        # Reversed cost matrix compared to AEC!
        logaec = y * (np.log(scores + eps) * cost_matrix[:, 0, 1] + np.log(1 - scores + eps) * cost_matrix[:, 1, 1]) \
            + (1 - y) * (np.log(1 - scores + eps) * cost_matrix[:, 1, 0] + np.log(scores + eps) * cost_matrix[:, 0, 0])

        logaec = - logaec  # Negative because of the logarithm

        # print(logaec.mean())

        # Add regularization
        logaec += self.lambda1 * np.sum(abs(theta[1:])) + self.lambda2 * np.sum(theta[1:] ** 2)

        return logaec.mean()

    def predict(self, x_predict):

        # scores = 1 / (1 + np.exp(-self.theta[0] - x_predict.dot(self.theta[1:])))    #eq 9
        scores = expit(self.theta[0] + x_predict.dot(self.theta[1:]))

        return scores

    def tune(self, l1, lambda1_list, l2, lambda2_list, x_train, y_train, cost_matrix_train, x_val, y_val,
             cost_matrix_val):
        if l1:
            self.lambda2 = 0
            losses_list = []
            for lambda1 in lambda1_list:
                logit = CSLogit(self.initial_theta, lambda1=lambda1, lambda2=0, obj=self.obj)
                logit.fitting(x_train, y_train, cost_matrix_train)

                # Evaluate loss (without regularization term!)
                logit.lambda1 = 0
                if self.obj == 'ce':
                    val_loss = self.ce(logit.theta_opt, x_val, y_val)
                elif self.obj == 'weightedce':
                    misclass_cost_val = np.zeros(len(y_val))
                    misclass_cost_val[y_val == 0] = cost_matrix_val[:, 1, 0][y_val == 0]
                    misclass_cost_val[y_val == 1] = cost_matrix_val[:, 0, 1][y_val == 1]

                    val_loss = self.weighted_ce(logit.theta_opt, x_val, y_val, misclass_cost_val)
                elif self.obj == 'aec':
                    val_loss = self.aec(logit.theta_opt, x_val, y_val, cost_matrix_val)
                elif self.obj == 'logaec':
                    val_loss = self.logaec(logit.theta_opt, x_val, y_val, cost_matrix_val)
                print('\t\tLambda l1 = %.5f;\tLoss = %.5f' % (lambda1, val_loss))
                losses_list.append(val_loss)
            lambda1_opt = lambda1_list[np.argmin(losses_list)]
            print('\tOptimal lambda = %.5f' % lambda1_opt)
            self.lambda1 = lambda1_opt

        elif l2:
            self.lambda1 = 0
            losses_list = []
            for lambda2 in lambda2_list:
                logit = CSLogit(self.initial_theta, lambda1=0, lambda2=lambda2, obj=self.obj)
                logit.fitting(x_train, y_train, cost_matrix_train)

                # Evaluate loss (without regularization term!)
                logit.lambda2 = 0
                if self.obj == 'ce':
                    val_loss = self.ce(logit.theta_opt, x_val, y_val)
                elif self.obj == 'weightedce':
                    misclass_cost_val = np.zeros(len(y_val))
                    misclass_cost_val[y_val == 0] = cost_matrix_val[:, 1, 0][y_val == 0]
                    misclass_cost_val[y_val == 1] = cost_matrix_val[:, 0, 1][y_val == 1]

                    val_loss = self.weighted_ce(logit.theta_opt, x_val, y_val, misclass_cost_val)
                elif self.obj == 'aec':
                    val_loss = self.aec(logit.theta_opt, x_val, y_val, cost_matrix_val)
                elif self.obj == 'logaec':
                    val_loss = self.logaec(logit.theta_opt, x_val, y_val, cost_matrix_val)
                print('\t\tLambda l1 = %.5f;\tLoss %.5f' % (lambda2, val_loss))
                losses_list.append(val_loss)
            lambda2_opt = lambda2_list[np.argmin(losses_list)]
            print('\tOptimal lambda = %.5f' % lambda2_opt)
            self.lambda2 = lambda2_opt
        else:
            self.lambda1 = 0
            self.lambda2 = 0
