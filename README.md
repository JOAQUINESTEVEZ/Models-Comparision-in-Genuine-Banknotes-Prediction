# Models Comparison in Genuine Banknotes Prediction
// 👨‍💻Author: Joaquin Estevez Year: 2023

## 🎯Project Goal 
The goal of this project is to compare the performance of different machine learning models on the task of predicting whether a banknote is authentic or counterfeit. The project aims to analyze and evaluate various models to determine which one provides better outcomes for the given prediction problem.


## 🌐Overview 
  The project employs four different machine learning models:
  1. ### Perceptron Model:
     - Simple linear binary classification algorithm. It learns a linear decision boundary to separate two classes based on input features.
  2. ### SVC Model:
     - Support Vector Machine (SVM) is a powerful supervised learning algorithm for classification and regression tasks. SVC is the classification variant of SVM. It works by finding the optimal hyperplane that maximally separates the data points of different classes. It is effective in high-dimensional spaces and can handle non-linear decision boundaries through the use of kernels.
  3. ### K-Nearest Neighbors (KNN) Model:
     - KNN is a non-parametric, lazy learning algorithm used for classification and regression tasks. In KNN, a data point is classified by the majority class of its k nearest neighbors. It does not make assumptions about the underlying data distribution and can adapt to complex decision boundaries.
  4. ### Gaussian Naive Bayes Model:
     -  Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem. The Gaussian Naive Bayes assumes that features follow a Gaussian (normal) distribution. It calculates the probability of each class for a given set of features and selects the class with the highest probability. Despite its "naive" assumption of feature independence, it often performs well in practice and is computationally efficient.

  ```python
  perceptron_model = Perceptron()
  svc_model = svm.SVC()
  knc_model = KNeighborsClassifier(n_neighbors=2)
  gaussianNB_model = GaussianNB()
  ```
  - ### CSV
      - The CSV file has 5 columns: `variance`, `skewness`, `curtosis`,	`entropy`, and `class`.
      - The first four columns serve as characteristics that help us find patters.
      - The last column is the label. It indicates if a banknote is authentic or counterfeit.
        
      ```csv
      variance	skewness	curtosis	entropy	    class
      -0.89569	3.0025	        -3.6067	        -3.4457	    1
      ```

## 🔧Usage 
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


## 📈Results 
  ```txt
    Results for model Perceptron
    Correct: 544
    Incorrect: 5
    Accuracy: 99.09%
  ```
  ```txt
    Results for model SVC
    Correct: 548
    Incorrect: 1
    Accuracy: 99.82%
  ```
  ```txt
    Results for model KNeighborsClassifier
    Correct: 549
    Incorrect: 0
    Accuracy: 100.00%
  ```
  ```txt
    Results for model GaussianNB
    Correct: 463
    Incorrect: 86
    Accuracy: 84.34%
  ```

##  🧠Conclusion and Analysis
  1. The results demonstrate that the Perceptron model achieved a commendable accuracy of 99.09%, correctly classifying 544 out of 549 instances. Similarly, the Support Vector Classifier (SVC) exhibited outstanding performance, with an accuracy of 99.82%, making only 1 incorrect prediction out of 549. The K-Nearest Neighbors (KNN) model showcased an impeccable accuracy of 100%, achieving correct predictions for all instances.

  2. In contrast, the Gaussian Naive Bayes model, while still performing reasonably well, showed a lower accuracy of 84.34%. This discrepancy suggests that the assumptions of feature independence made by the Gaussian Naive Bayes algorithm might not align perfectly with the underlying distribution of the banknote characteristics.
  3. The Perceptron, SVC, and KNN models are versatile and can adapt to complex decision boundaries. They can capture intricate patterns and relationships in the data.
  4. Gaussian Naive Bayes assumes feature independence and follows a probabilistic approach. This may be less suitable for datasets where features are not entirely independent.
  5. The dataset might contain non-linear relationships between features and the authenticity of banknotes. Support Vector Classifier (SVC) is explicitly designed to handle non-linear decision boundaries through the use of kernels, making it effective in capturing complex patterns that might exist in the data.



