import csv
import random

from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

perceptron_model = Perceptron()
svc_model = svm.SVC()
knc_model = KNeighborsClassifier(n_neighbors=2)
gaussianNB_model = GaussianNB()

models = [perceptron_model, svc_model, knc_model, gaussianNB_model]

# Read data in from file
with open("banknotes.csv") as f:
    reader = csv.reader(f)
    next(reader)

    data = []
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": "Authentic" if row[4] == "0" else "Counterfeit"
        })

# Separate data into training and testing groups
evidence = [row["evidence"] for row in data]
labels = [row["label"] for row in data]

X_training, X_testing, y_training, y_testing = train_test_split(
    evidence, labels, test_size=0.4
)

# Fit models
for model in models:
    model.fit(X_training, y_training)

# Make predictions on the testing set
predictions = []
for model in models:
    prediction = model.predict(X_testing)
    predictions.append(prediction)

# Compute how well we performed
results = []
for prediction in predictions:
    correct = (y_testing == prediction).sum()
    incorrect = (y_testing != prediction).sum()
    total = len(prediction)
    results.append([correct, incorrect, total])

# Print results
for i in range(len(models)):
    print(f"Results for model {type(models[i]).__name__}")
    print(f"Correct: {results[i][0]}")
    print(f"Incorrect: {results[i][1]}")
    print(f"Accuracy: {100 * results[i][0] / results[i][2]:.2f}%")
    print()