'''
This program runs a set of classifiers to determine predictions outcomes.
A dataset of ECG features for multiple patients are used to score the prediction models.
'''


# Importing pandas and numpy for data processing and overall coding
import pandas as pd
import numpy as np

# Importing libraries for data splitting, feature selection, different classifiers,
# and classification metrices
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, QuantileTransformer  # --> QuantileTransformer unused?
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix  # --> unused?
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold

# Importing the load_data function from the ecg module
from ecg.load_data import load_data


# --------------
# Data importing
# Importing the ECG features dataset
# --------------

data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')


# --------------
# Data splitting
# Data is split in training and test set, where the training set is 80% of the total dataset.
# Split is stratified based on the given labels.
# --------------

labels = data.pop('label')
x, x_test, y, y_test = train_test_split(data, labels, test_size=0.2, train_size=0.8, stratify=labels)


# ---------------
# Feature scaling
# The features are scaled using RobustScalar.
# ---------------

scaler = RobustScaler()
scaler.fit_transform(x)


# ----------------------------------
# Principal Component Analysis (PCA)
# Performing the PCA with a total number of components where the accumulated variance
# sums up to at least 90%.
# ----------------------------------

pca = PCA(n_components=0.90)
principal_components_train = pca.fit_transform(x)
principal_components_test = pca.transform(x_test)

x = pd.DataFrame(data=principal_components_train)
x_test = pd.DataFrame(data=principal_components_test)

assert all(x) <= 1


# ----------
# Classifier
# A function is created to test and run multiple classifiers for the given data
# ----------

def classifier(x_train, x_test, y_train, y_test):
    '''
    This function defines multiple classifiers.
    All classifiers are created, fitted, and the predictions are captured.

    arg1 = x_train, the training data
    arg2 = x_test, the test data
    arg3 = y_train, the training labels
    arg4 = y_test, the test labels

    return:
    predictions, predictions
    pred_accuracies, accurary scores
    pred_metrics, multiple scoring values
    '''

    svc_model = SVC()
    knn_model = KNeighborsClassifier(n_neighbors=10)
    lg_model = LogisticRegression(max_iter=10000)
    dtr_model = DecisionTreeClassifier()
    rfc_model = RandomForestClassifier()
    gnb_model = GaussianNB()

    svc_model.fit(x_train, y_train)
    knn_model.fit(x_train, y_train)
    lg_model.fit(x_train, y_train)
    dtr_model.fit(x_train, y_train)
    rfc_model.fit(x_train, y_train)
    gnb_model.fit(x_train, y_train)

    predictions = {}
    predictions['SVC_prediction'] = svc_model.predict(x_test)
    predictions['KNN_prediction'] = knn_model.predict(x_test)
    predictions['LG_prediction'] = lg_model.predict(x_test)
    predictions['DTR_prediction'] = dtr_model.predict(x_test)
    predictions['RFC_prediction'] = rfc_model.predict(x_test)
    predictions['GNB_prediction'] = gnb_model.predict(x_test)

    pred_accuracies = {}
    for pred in predictions:
        pred_accuracies[pred] = accuracy_score(predictions[pred], y_test)

    pred_metrics = {}
    for pred in predictions:
        pred_metrics[pred] = classification_report(predictions[pred], y_test, zero_division=0)

    return predictions, pred_accuracies, pred_metrics


# -----------------------
# K-fold Cross-validation
# K-fold cross-validation is performed to check for generalization performance of the classifiers.
# -----------------------

k = 10
skf = StratifiedKFold(n_splits=k, shuffle=True)
all_pred_accuracies = {}
for train_index, test_index in skf.split(x, y):
    [predictions, pred_accuracies, pred_metrics] = classifier(x.iloc[train_index],
                                                              x.iloc[test_index],
                                                              y.iloc[train_index],
                                                              y.iloc[test_index])

    if all_pred_accuracies == {}:  # Initialize the dict that's going to hold all predictions
        all_pred_accuracies = pred_accuracies.copy()
        for pred_type in pred_accuracies.keys():
            # Convert dict items to list
            all_pred_accuracies[pred_type] = [all_pred_accuracies[pred_type]]
    else:
        for pred_type in pred_accuracies.keys():
            # Add accuracy scores to all_predictions dict
            all_pred_accuracies[pred_type].append(pred_accuracies[pred_type])

# Print average prediction accuracies
print(f'Average {k}-fold prediction accuracies:')
for pred_type in all_pred_accuracies:
    print(f'{pred_type}: {np.mean(all_pred_accuracies[pred_type])}')

# --------------------------
# Final test on test dataset
# --------------------------
RUN_FINAL_TEST = False
if RUN_FINAL_TEST:
    [predictions, pred_accuracies, pred_metrics] = classifier(x, x_test, y, y_test)

    for prediction in pred_accuracies:
        print(f'{prediction}: {pred_accuracies[prediction]}')

    for prediction in pred_metrics:
        print(f'{prediction}: {pred_metrics[prediction]}')
