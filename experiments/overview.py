# Global overview for all experiments
import os
import json
import datetime
from experiments import experiment

# Set project directory:
DIR = r'C:\Users\devos\PycharmProjects\RobustIDCS2022\results'

#Select models you want to train - By default set on True for all three
methodologies = {'logit': True,
                 'cslogit': True,
                 'cslogit_robust': True,
                 }

datasets = {'kaggle credit card fraud': True,
            'outlier_size': 1000000,  #Specify the amount of the outlier
            'outlier_index': 0
            }

#Set 'toy_example' to False if you do not want to test the models on synthetic data
toyexample = {'toy_example': True,
              'sample_n': 100,
              'vertical_split': 12,
              'basecost_neg': 20,
              'basecost_pos': 20,
              'cost_neg_x1_coeff': -2.5,
              'cost_pos_x1_coeff': 2.5,
              'random_seed': 0,
              'noise_factor': 5,
              'test_size': 0.25,
              'x1_outlier': -2.5,
              'x2_outlier': -2.5,
              'outlier_amount': 400,
              'noise_level': 1.5
              }

evaluators = {
              # Cost-insensitive metrics
              'traditional': True,
              'AUC': True,
              'PR': True,
              'recall_overlap': True,

              # Cost-sensitive metrics
              'savings': True,
              'AEC': True,
              }

# Specify experimental configuration
# l1 and l2 not supported simultaneously!
settings = {'class_costs': False,
            'folds': 5,
            'repeats': 2,
            'val_ratio': 0.25,
            'l1_regularization': False,
            'lambda1_options': [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'l2_regularization': False,
            'lambda2_options': [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'neurons_options': [2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10]
            }

if __name__ == '__main__':
    print('\n' + datetime.datetime.now().strftime('Experiment started at:  %d-%m-%y  |  %H:%M'))

    experiment = experiment.Experiment(settings, datasets, methodologies, toyexample, evaluators)
    experiment.run(directory=DIR)

    # Create txt file for summary of results
    with open(str(DIR + 'summary.txt'), 'w') as file:
        file.write(str(datetime.datetime.now().strftime('Experiment done at:  %d-%m-%y  |  %H:%M') + '\n'))
        file.write('\nSettings: ')
        file.write(json.dumps(settings, indent=3))
        file.write('\nDatasets: ')
        file.write(json.dumps(datasets, indent=3))
        file.write('\nMethodologies: ')
        file.write(json.dumps(methodologies, indent=3))
        file.write('\nEvaluators: ')
        file.write(json.dumps(evaluators, indent=3))
        file.write('\n\n_____________________________________________________________________\n\n')

    experiment.evaluate(directory=DIR)
