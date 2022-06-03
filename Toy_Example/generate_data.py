import numpy as np

def generate_data(sample_n, vertical_split, basecost_neg, basecost_pos, cost_neg_x1_coeff, cost_pos_x1_coeff,
                  random_seed, noise_factor):

    np.random.seed(random_seed)

    # Generate negative samples
    x1_neg = noise_factor * np.random.standard_normal(sample_n) * 1.2
    x2_neg = -vertical_split/2 + noise_factor * np.random.standard_normal(sample_n) * 1.2
    x_neg = np.transpose([x1_neg, x2_neg])
    cost_neg = (cost_neg_x1_coeff * x1_neg) + basecost_neg

    cost_neg[cost_neg < 0] = 0
    labels_neg = np.zeros(sample_n)

    # Generate positive samples
    x1_pos = noise_factor * np.random.standard_normal(sample_n) * 1.2
    x2_pos = vertical_split / 2 + noise_factor * np.random.standard_normal(sample_n) * 1.2
    x_pos = np.transpose([x1_pos, x2_pos])
    cost_pos = (cost_pos_x1_coeff * x1_pos) + basecost_pos

    cost_pos[cost_pos < 0] = 0
    labels_pos = np.ones(sample_n)

    # combine pos and neg generated data: X, labels, costs
    X = np.concatenate((x_pos, x_neg))
    labels = np.concatenate((labels_pos, labels_neg))
    costs = np.concatenate((cost_pos, cost_neg))

    return X, labels, costs,x_neg,x_pos,cost_neg,cost_pos