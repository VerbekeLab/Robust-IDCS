# Robust Instance-Dependent Cost-Sensitive Classification </br><sub><sub>Simon De Vos, Toon Vanderschueren, Tim Verdonck, and Wouter Verbeke[[2023]](https://doi.org/10.1007/s11634-022-00533-3)</sub></sub>

## Description
This is the code for the [paper](https://link.springer.com/article/10.1007/s11634-022-00533-3) on "Robust Instance-Dependent Cost-Sensitive Classification".

Contact the [author](https://www.kuleuven.be/wieiswie/nl/person/00148775) at simon.devos@kuleuven.be.

## Instructions
### Data:
The creditcard transaction data can be found here: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud \
The .csv file should be placed in the data folder as "data\Kaggle Creditcard Fraud\creditcard.csv". You can replace the now empty creditcard.csv file.
### Run code:
Run overview.py to execute the experiments as described in the paper. \
Settings can be adapted in overview.py: 
 * Set DIR variable to your custom result folder 
 * Specify experimental configuration 
 * Default settings:  
     5-fold cross-validation, 2 repeats \
     Toy example on synthetic data is generated and displayed \
     Three classifiers are trained: logit, cslogit, r-cslogit \
     evaluators: traditional, AUC, Savings 

## Acknowledgments
The code for cslogit is a Python version of the [original cslogit by Sebastiaan Höppner et al.](https://github.com/SebastiaanHoppner/CostSensitiveLearning).

## Citing
Please cite our paper and/or code as follows:

```tex

De Vos, Simon, Toon Vanderschueren, Tim Verdonck, and Wouter Verbeke. 2023. “Robust Instance-Dependent Cost-Sensitive Classification.” Advances in Data Analysis and Classification, January. https://doi.org/10.1007/s11634-022-00533-3.


```