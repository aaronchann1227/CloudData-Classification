# CloudData-Classification
Exploration and modeling of cloud detection in the polar regions based on radiance recorded automatically by the MISR sensor abroad the NASA satellite Terra


#CVGeneric (the cross validation function)
this function performs K-fold cross validation on a classifier.
Function parameters:
  classifier: string to identify the classifier. Options include:
    "QDA" - Quadratic Discriminant Analysis
    "LDA" - Linear Discriminant Analysis
    "logistic" - Logistic Regression
    "kernelSVM" - Kernel SVM (RBF kernel)
    "dtree" - Decision tree
  data: the data to train on
  K: the amount of folds
  loss: the loss function
  hyperparameters: hyperparameters for a model formatted as a list
    "C" and "sigma" are for kernel SVM
    "prior" is for QDA/LDA
  formula: string formula
  splitMethod: method of splitting (integer)
    1: splitting by blocks
    2: splitting by image