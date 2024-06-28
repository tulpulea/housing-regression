Project 3 Writeup

Data Source: https://www.kaggle.com/datasets/sukhmandeepsinghbrar/housing-price-dataset

The following is the write-up for this personal project completed by Archan Tulpule.

Project Description: Ridge Regression vs BART prediction performance on housing dataset.

Skills demonstrated: Python, Machine learning, Regression, Bias-Variance Tradeoff, Training vs Testing error, Ensemble learning, Regularization, K-fold Cross-validation, pre-processing

Solution description: 

In this project I tackle a hosuing data set which has numerous predictors for a quantitative response - price. Features include aspects of housing such as square feet, zip code, date of construction, view, number of floors, etc. 

The aim of this project was to compare and contrast two approaches to this regression problem.

1) A ridge regression approach, ie: multiple linear regression with L2 regularization. The power of this apporach is its simplicity and effectiveness in reducing variance and thereofore increasing the accuracy of predictions. To ensure a suitable value for the shrinkage paramter was chosen, 5-fold cross-validation was done with an array of different paramter values.

2) An ensemble learning approach, Bayesian Additive Regression Trees (BART). The strength of this method arises from its Bayesian approach to model fitting, whereby some K number of trees are updated over some B iterations with perturbations made from the previous iteration's tree following a posterior distribution. This model is non-parametric and relies on summing up the K trees at each iteration which are treated as "weak learners" and then a final model is based off of the mean of those B models. 

The loss fucntion used to measure fit was the root mean square error, and a baseline model (mean of reponse) was used as a benchmark.  

The code is contained in main.ipynb, which goes through all the steps done in the project, a shortened version with the main code is also available on main.py. 