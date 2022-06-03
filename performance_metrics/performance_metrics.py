import numpy as np
#import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.stats import spearmanr, combine_pvalues, friedmanchisquare
#from scikit_posthocs import posthoc_nemenyi_friedman
from tabulate import tabulate
#import Orange
#from Orange.evaluation import compute_CD, graph_ranks
#from hmeasure import h_score
#import os
#import baycomp

def savings(cost_matrix, labels, predictions):
    cost_without = cost_without_algorithm(cost_matrix, labels)
    cost_with = cost_with_algorithm(cost_matrix, labels, predictions)
    savings = 1 - cost_with / cost_without

    return savings

def cost_with_algorithm(cost_matrix, labels, predictions):

    cost_tn = cost_matrix[:, 0, 0][np.logical_and(predictions == 0, labels == 0)].sum()
    cost_fn = cost_matrix[:, 0, 1][np.logical_and(predictions == 0, labels == 1)].sum()
    cost_fp = cost_matrix[:, 1, 0][np.logical_and(predictions == 1, labels == 0)].sum()
    cost_tp = cost_matrix[:, 1, 1][np.logical_and(predictions == 1, labels == 1)].sum()

    return sum((cost_tn, cost_fn, cost_fp, cost_tp))

def cost_without_algorithm(cost_matrix, labels):

    # Predict everything as the default class that leads to minimal cost
    # Also include cost of TP/TN!
    cost_neg = cost_matrix[:, 0, 0][labels == 0].sum() + cost_matrix[:, 0, 1][labels == 1].sum()
    cost_pos = cost_matrix[:, 1, 0][labels == 0].sum() + cost_matrix[:, 1, 1][labels == 1].sum()

    return min(cost_neg, cost_pos)

def get_performance_metrics(evaluators, evaluation_matrices, i, index, cost_matrix, labels, probabilities, predictions,
                            info):
    if evaluators['traditional']:
        true_pos = (predictions * labels).sum()
        true_neg = ((1-predictions) * (1-labels)).sum()
        false_pos = (predictions * (1-labels)).sum()
        false_neg = ((1-predictions) * labels).sum()

        accuracy = (true_pos + true_neg) / len(labels)
        recall = true_pos / (true_pos + false_neg)
        # Make sure no division by 0!
        if (true_pos == 0) and (false_pos == 0):
            precision = 0
            print('\t\tWARNING: No positive predictions!')
        else:
            precision = true_pos / (true_pos + false_pos)
        if precision == 0:
            f1_score = 0
            print('\t\tWARNING: Precision = 0!')
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        evaluation_matrices['traditional'][index, i] = np.array([accuracy, recall, precision, f1_score])

    if evaluators['AUC']:
        auc = metrics.roc_auc_score(y_true=labels, y_score=probabilities)
#        print('\t\tAUC:\t\t ' + str(round(auc,4)))
        evaluation_matrices['AUC'][index, i] = auc

    if evaluators['savings']:
        # To do: function - savings
        cost_without = cost_without_algorithm(cost_matrix, labels)
        cost_with = cost_with_algorithm(cost_matrix, labels, predictions)
        savings = 1 - cost_with / cost_without
#        print('\t\tsavings:\t ' + str(round(savings,4)))
        evaluation_matrices['savings'][index, i] = savings

    if evaluators['AEC']:
        expected_cost = labels * (probabilities * cost_matrix[:, 1, 1] + (1 - probabilities) * cost_matrix[:, 0, 1]) \
            + (1 - labels) * (probabilities * cost_matrix[:, 1, 0] + (1 - probabilities) * cost_matrix[:, 0, 0])

        aec = expected_cost.mean()
#        print('\t\t\tAEC:\t ' + str(round(aec,4)))
        evaluation_matrices['AEC'][index, i] = aec

    if evaluators['PR']:
        precision, recall, _ = metrics.precision_recall_curve(y_true=labels, probas_pred=probabilities)

        # AUC is not recommended here (see sklearn docs)
        # We will use Average Precision (AP)
        ap = metrics.average_precision_score(y_true=labels, y_score=probabilities)

        evaluation_matrices['PR'][index, i] = np.array([precision, recall, ap], dtype=object)

    if evaluators['recall_overlap']:

        recalled = labels[labels == 1] * predictions[labels == 1]

        evaluation_matrices['recall_overlap'][index, i] = recalled
    """
    if evaluators['recall_correlation']:

        pos_probas = probabilities[labels == 1]

        # Sort indices from high to low
        sorted_indices_probas = np.argsort(pos_probas)[::-1]
        prob_rankings = np.argsort(sorted_indices_probas)

        evaluation_matrices['recall_correlation'][index, i] = prob_rankings
    """
    return evaluation_matrices


def evaluate_experiments(evaluators, methodologies, evaluation_matrices, directory, name):
    table_evaluation = []
    n_methodologies = sum(methodologies.values())

    names = []
    for key in methodologies.keys():
        if methodologies[key]:
            names.append(key)

    if evaluators['traditional']:

        table_traditional = [['Method', 'Accuracy', 'Recall', 'Precision', 'F1-score', 'AR', 'sd']]

        # Compute F1 rankings (- as higher is better)
        all_f1s = []
        for i in range(evaluation_matrices['traditional'].shape[0]):
            method_f1s = []
            for j in range(evaluation_matrices['traditional'][i].shape[0]):
                f1 = evaluation_matrices['traditional'][i][j][-1]
                method_f1s.append(f1)
            all_f1s.append(np.array(method_f1s))

        ranked_args = np.argsort(-np.array(all_f1s), axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = np.mean(rankings, axis=1)
        sd_rankings = np.sqrt(rankings.var(axis=1))

        # Summarize all per method
        index = 0
        for item, value in methodologies.items():
            if value:
                averages = evaluation_matrices['traditional'][index, :].mean()

                table_traditional.append([item, averages[0], averages[1], averages[2], averages[3],
                                          avg_rankings[index], sd_rankings[index]])

                index += 1

        print(tabulate(table_traditional, headers="firstrow", floatfmt=("", ".4f", ".4f", ".4f", ".4f", ".4f", ".4f")))
        table_evaluation.append(table_traditional)
        print('_________________________________________________________________________')

    if evaluators['AUC']:

        table_auc = [['Method', 'AUC', 'sd', 'AR', 'sd']]

        # Compute rankings (- as higher is better)
        ranked_args = (-evaluation_matrices['AUC']).argsort(axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = rankings.mean(axis=1)
        sd_rankings = np.sqrt(rankings.var(axis=1))

        # Summarize per method
        index = 0
        for item, value in methodologies.items():
            if value:
                table_auc.append([item, evaluation_matrices['AUC'][index, :].mean(),
                                  np.sqrt(evaluation_matrices['AUC'][index, :].var()), avg_rankings[index],
                                  sd_rankings[index]])
                index += 1

        print(tabulate(table_auc, headers="firstrow", floatfmt=("", ".4f", ".4f", ".4f", ".4f")))
        table_evaluation.append(table_auc)

        print('_________________________________________________________________________')

    if evaluators['savings']:

        table_savings = [['Method', 'Savings', 'sd', 'AR', 'sd']]

        # Compute rankings (- as higher is better)
        ranked_args = (-evaluation_matrices['savings']).argsort(axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = rankings.mean(axis=1)
        sd_rankings = np.sqrt(rankings.var(axis=1))

        # Summarize per method
        index = 0
        methods_used = []
        for item, value in methodologies.items():
            if value:
                methods_used.append(item)
                table_savings.append([item, evaluation_matrices['savings'][index, :].mean(),
                                      np.sqrt(evaluation_matrices['savings'][index, :].var()), avg_rankings[index],
                                      sd_rankings[index]])
                index += 1

        print(tabulate(table_savings, headers="firstrow", floatfmt=("", ".4f", ".4f", ".4f", ".4f")))
        table_evaluation.append(table_savings)
        print('_________________________________________________________________________')

    if evaluators['AEC']:

        table_aec = [['Method', 'AEC', 'sd', 'AR', 'sd']]

        # Compute rankings (lower is better)
        ranked_args = (evaluation_matrices['AEC']).argsort(axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = rankings.mean(axis=1)
        sd_rankings = np.sqrt(rankings.var(axis=1))

        # Summarize per method
        index = 0
        methods_used = []
        for item, value in methodologies.items():
            if value:
                methods_used.append(item)
                table_aec.append([item, evaluation_matrices['AEC'][index, :].mean(),
                                  np.sqrt(evaluation_matrices['AEC'][index, :].var()), avg_rankings[index],
                                  sd_rankings[index]])
                index += 1

        print(tabulate(table_aec, headers="firstrow", floatfmt=("", ".4f", ".4f", ".4f", ".4f")))
        table_evaluation.append(table_aec)

        print('_________________________________________________________________________')


    if evaluators['PR']:
        table_ap = [['Method', 'Avg Prec', 'sd', 'AR', 'sd']]

        index = 0

        all_aps = []
        for item, value in methodologies.items():
            if value:
                # See https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
                precisions = []
                mean_recall = np.linspace(0, 1, 100)
                aps = []

                for i in range(evaluation_matrices['PR'][index, :].shape[0]):
                    precision, recall, ap = list(evaluation_matrices['PR'][index, i])

                    interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
                    interp_precision[0] = 1
                    precisions.append(interp_precision)

                    aps.append(ap)

                mean_precision = np.mean(precisions, axis=0)
                mean_precision[-1] = 0

                aps = np.array(aps)
                table_ap.append([item, aps.mean(), np.sqrt(aps.var())])

                all_aps.append(aps)

                index += 1

        # Add rankings (higher is better)
        ranked_args = np.argsort(-np.array(all_aps), axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = np.mean(rankings, axis=1)
        sd_rankings = np.sqrt(rankings.var(axis=1))
        for i in range(1, len(table_ap)):
            table_ap[i].append(avg_rankings[i - 1])
            table_ap[i].append(sd_rankings[i - 1])

        print(tabulate(table_ap, headers="firstrow", floatfmt=("", ".4f", ".4f", ".4f", ".4f")))
        table_evaluation.append(table_ap)
        print('_________________________________________________________________________')

    if evaluators['recall_overlap']:

        # Make table with only relevant methodologies
        table_recall_overlap = ['Recall overlaps']
        for meth in methodologies:
            if methodologies[meth]:
                table_recall_overlap.append(meth)
        table_recall_overlap = [table_recall_overlap]

        # Get recall overlap per experiment (fold/repeat)
        n_experiments = evaluation_matrices['recall_overlap'].shape[1]
        recall_overlaps = np.zeros((n_experiments, n_methodologies, n_methodologies))
        for n in range(n_experiments):
            for i in range(n_methodologies):
                for j in range(n_methodologies):
                    # if j > i:
                    #    break
                    recall_overlaps[n, i, j] = (evaluation_matrices['recall_overlap'][i, n] == evaluation_matrices['recall_overlap'][j, n]).mean()

        # Summarize over repeated experiments
        recall_overlaps = recall_overlaps.mean(axis=0)

        # Summarize per method
        index = 0
        for item, value in methodologies.items():
            if value:
                table_recall_overlap.append([item] + list(recall_overlaps[index, :]))
                index += 1

        print(tabulate(table_recall_overlap, headers="firstrow", floatfmt=()))
        table_evaluation.append(table_recall_overlap)

        print('_________________________________________________________________________')
    """
    if evaluators['recall_correlation']:

        # Make table with only relevant methodologies
        table_recall_correlations = ['Recall correlations']
        for meth in methodologies:
            if methodologies[meth]:
                table_recall_correlations.append(meth)
        table_recall_correlations = [table_recall_correlations]

        # Get recall correlation per experiment (fold/repeat)
        n_experiments = evaluation_matrices['recall_correlation'].shape[1]
        recall_correlations = np.zeros((n_experiments, n_methodologies, n_methodologies))
        for n in range(n_experiments):
            for i in range(n_methodologies):
                for j in range(n_methodologies):
                    # if j > i:
                    #    break
                    # Todo: Spearman's correlation R
                    spearman_corr = spearmanr(evaluation_matrices['recall_correlation'][i, n],
                                                             evaluation_matrices['recall_correlation'][j, n])

                    recall_correlations[n, i, j] = spearman_corr[0]

        # Summarize over repeated experiments
        recall_correlations = recall_correlations.mean(axis=0)

        # Summarize per method
        index = 0
        for item, value in methodologies.items():
            if value:
                table_recall_correlations.append([item] + list(recall_correlations[index, :]))
                index += 1

        print(tabulate(table_recall_correlations, headers="firstrow", floatfmt=()))
        table_evaluation.append(table_recall_correlations)

        print('_________________________________________________________________________')
    """
