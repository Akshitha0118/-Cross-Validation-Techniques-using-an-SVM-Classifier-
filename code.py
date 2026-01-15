# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# read the dataset
dataset = pd.read_csv(r"C:\Users\Admin\Desktop\cross validation svc\Social_Network_Ads.csv")


# divide x and y variables (independent & dependent variable)
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values


# import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)


# feature scalling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# model building 
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)


# model predictions
y_pred = classifier.predict(X_test)


# confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)


# accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# bias & variance 
bias = classifier.score(X_train, y_train)
variance = classifier.score(X_test, y_test)

print("Training Accuracy (Bias):", bias)
print("Testing Accuracy (Variance):", variance)


# cross_val_score
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(
    estimator=classifier,
    X=X_train,
    y=y_train,
    cv=8
)
print("Cross Validation Accuracy: {:.2f} %".format(accuracies.mean() * 100))


# GridSearch cross validation
from sklearn.model_selection import GridSearchCV

parameters = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'kernel': ['rbf']}
]

grid_search = GridSearchCV(
    estimator=classifier,
    param_grid=parameters,
    scoring='accuracy',
    cv=10,
    n_jobs=-1
)

grid_search = grid_search.fit(X_train, y_train)

print("Best GridSearch Accuracy:", grid_search.best_score_)
print("Best Parameters:", grid_search.best_params_)


# RandomSearch cross validation
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

parameters_random = {
    'C': uniform(1, 1000),
    'kernel': ['linear', 'rbf'],
    'gamma': uniform(0.01, 1)
}

random_search = RandomizedSearchCV(
    estimator=classifier,
    param_distributions=parameters_random,
    n_iter=50,
    scoring='accuracy',
    cv=10,
    random_state=0,
    n_jobs=-1
)

random_search = random_search.fit(X_train, y_train)

print("Best RandomizedSearch Accuracy:", random_search.best_score_)
print("Best Parameters:", random_search.best_params_)


