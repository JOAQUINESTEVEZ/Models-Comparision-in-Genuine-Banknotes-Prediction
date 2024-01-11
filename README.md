# Models Comparison in Genuine Banknotes Prediction
// Author: Joaquin Estevez Year: 2023

## Project Goal
The goal of this project is to compare the performance of different machine learning models on the task of predicting whether a banknote is authentic or counterfeit. The project aims to analyze and evaluate various models to determine which one provides better outcomes for the given prediction problem.


## Overview
  The project employs four different machine learning models:
  1. ### Perceptron Model:
     - Simple linear binary classification algorithm. It learns a linear decision boundary to separate two classes based on input features.
  2. ### SVC Model:
     - Support Vector Machine (SVM) is a powerful supervised learning algorithm for classification and regression tasks. SVC is the classification variant of SVM. It works by finding the optimal hyperplane that maximally separates the data points of different classes. It is effective in high-dimensional spaces and can handle non-linear decision boundaries through the use of kernels.
  3. ### K-Nearest Neighbors (KNN) Model:
     - KNN is a non-parametric, lazy learning algorithm used for classification and regression tasks. In KNN, a data point is classified by the majority class of its k nearest neighbors. It does not make assumptions about the underlying data distribution and can adapt to complex decision boundaries.
  4. ### Gaussian Naive Bayes Model:
     -  Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem. The Gaussian Naive Bayes assumes that features follow a Gaussian (normal) distribution. It calculates the probability of each class for a given set of features and selects the class with the highest probability. Despite its "naive" assumption of feature independence, it often performs well in practice and is computationally efficient.
## Usage
  The project is implemented in Python, and the main script is `predictions.py`. This script performs the following steps:
  1. Reads data from the "banknotes.csv" file.
  2. Divides the data into training and testing sets.
  3. Fits the specified models using the training data.
  4. Makes predictions on the testing set using each model.
  5. Computes and prints the performance metrics for each model, including correct predictions, incorrect predictions, and accuracy.
  
  To run the script, ensure you have the required libraries installed. You can run the following command:
  ```bash
  pip install scikit-learn
  ```

