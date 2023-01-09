# Robust IDCS

## Description
This is the code for the paper on "Robust Instance-Dependent Cost-Sensitive Classification".

Citation: De Vos, Simon, Toon Vanderschueren, Tim Verdonck, and Wouter Verbeke. 2023. “Robust Instance-Dependent Cost-Sensitive Classification.” Advances in Data Analysis and Classification, January. https://doi.org/10.1007/s11634-022-00533-3.

Contact the [author](https://www.kuleuven.be/wieiswie/nl/person/00148775) at simon.devos@kuleuven.be.

## Instructions
### Data:
The creditcard transaction data can be found here: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud \
The .csv file should be placed in the data folder as "data\Kaggle Creditcard Fraud\creditcard.csv". You can replace the now empty creditcard.csv file.
### Run code:
Run overview.py to execute the experiments as descibed in the paper. \
Settings can be adapted in overview.py: 
 * Set DIR variable to your custom result folder 
 * Specify experimental configuration 
 * Default settings:  
     5-fold cross-validation, 2 repeats \
     Toy example on synthetic data is generated and displayed \
     Three classifiers are trained: logit, cslogit, r-cslogit \
     evaluators: traditional, AUC, Savings 

## Acknowledgments
The code for cslogit is a Python versions of the [original cslogit by Sebastiaan Höppner et al.](https://github.com/SebastiaanHoppner/CostSensitiveLearning).
A framework of the experiments was largely provided by [Toon Vanderschueren](https://www.kuleuven.be/wieiswie/nl/person/00140754).


