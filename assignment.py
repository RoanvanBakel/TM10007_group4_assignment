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
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.utils import parallel_backend
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Importing the load_data function from the ecg module
from ecg.load_data import load_data


# ----------------------------------
# Data importing
# ----------------------------------
# Importing the ECG features dataset
data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')


# -------------------------------------------------------------------------------------------
# Data splitting
# -------------------------------------------------------------------------------------------
# Data is split in training and test set, where the training set is 80% of the total dataset.
# Split is stratified based on the given labels.
labels = data.pop('label')
x, x_test, y, y_test = train_test_split(data, labels, test_size=0.2, train_size=0.8,
                                        stratify=labels)


# ------------------------------------------------------
# Upsampling
# ------------------------------------------------------
# Upsampling training data to achieve 50/50 label split.
def upsampler(x, y):
    df = pd.concat([x, y], axis=1)
    df_majority = df[df.label == 0]
    df_minority = df[df.label == 1]

    df_minority_upsampled = resample(df_minority,
                                    replace=True,
                                    n_samples=len(df_majority.index),
                                    random_state=123)

    x = pd.concat([df_majority, df_minority_upsampled])
    y = x.pop('label')

    return x, y

[x, y] = upsampler(x, y)


# ---------------
# Feature scaling
# ---------------
scaler = MinMaxScaler()
scaler.fit_transform(x)
scaler.transform(x_test)


# ----------------------------------------------------------------------------------
# Principal Component Analysis (PCA)
# -----------------------------------------------------------------------------------
# Performing the PCA with a total number of components where the accumulated variance
# sums up to at least 90%.
pca = PCA(n_components=0.95)
principal_components_train = pca.fit_transform(x)
principal_components_test = pca.transform(x_test)

x = pd.DataFrame(data=principal_components_train)
x_test = pd.DataFrame(data=principal_components_test)
y = y.values.tolist()
y = pd.DataFrame(data=y, columns=['label'])


# -----------------------------------------------------------------------------
# Classifier
# -----------------------------------------------------------------------------
# A function is created to test and run multiple classifiers for the given data

# Define classifier models
svc_model = SVC(C=10)
rfc_model = RandomForestClassifier(n_estimators=50)

def fit_classifier(x_train, x_val_test, y_train, y_val_test):
    '''
    This function defines multiple classifiers.
    All classifiers are created, fitted, and the predictions are captured.

    arg1 = x_train, the training data
    arg2 = x_val_test, the validation/test data
    arg3 = y_train, the training labels
    arg4 = y_val_test, the validation/test labels

    return:
    predictions, predictions
    pred_accuracies, accurary scores
    pred_metrics, multiple scoring values
    '''

    # Upsampling training data to achieve 50/50 label split
    [x_train, y_train] = upsampler(x_train, y_train)
    svc_model.fit(x_train, y_train)
    rfc_model.fit(x_train, y_train)

    predictions = {}
    predictions['SVC_prediction'] = svc_model.predict(x_val_test)
    predictions['RFC_prediction'] = rfc_model.predict(x_val_test)

    pred_accuracies = {}
    for pred in predictions:
        pred_accuracies[pred] = accuracy_score(predictions[pred], y_val_test)

    pred_metrics = {}
    for pred in predictions:
        pred_metrics[pred] = classification_report(predictions[pred], y_val_test, zero_division=0)

    return predictions, pred_accuracies, pred_metrics


# ------------------------------------------------------------------------------------------------
# K-fold Cross-validation
# ------------------------------------------------------------------------------------------------
# K-fold cross-validation is performed to check for generalization performance of the classifiers.
k = 10
skf = StratifiedKFold(n_splits=k, shuffle=True)
all_pred_accuracies = {}
for train_index, test_index in skf.split(x, y):
    [predictions, pred_accuracies, pred_metrics] = fit_classifier(x.iloc[train_index],
                                                                  x.iloc[test_index],
                                                                  y.iloc[train_index],
                                                                  y.iloc[test_index])

    if all_pred_accuracies == {}:  # Initialize the dict that's going to hold all predictions
        all_pred_accuracies = pred_accuracies.copy()
        for pred_type in pred_accuracies:
            # Convert dict items to list
            all_pred_accuracies[pred_type] = [all_pred_accuracies[pred_type]]
    else:
        for pred_type in pred_accuracies:
            # Add accuracy scores to all_predictions dict
            all_pred_accuracies[pred_type].append(pred_accuracies[pred_type])

boxplt = pd.DataFrame(all_pred_accuracies)

sns.set(context='notebook', style='whitegrid', font_scale=2)


# Plot the graph
plot = sns.boxplot(data=boxplt, whis=np.inf, width=.18)
plot.set(title='Boxplot of accuracy for SVM and RFC after cross-validation',
         xlabel='Classifier', ylabel='Accuracy',)
plt.show()


print(f'Average {k}-fold prediction accuracies:')
for pred_type in all_pred_accuracies:
    print(f'{pred_type}: {np.mean(all_pred_accuracies[pred_type])}')


# ------------------------
# Grid Search Optimization
# ------------------------
run_grid_search = False
if run_grid_search:
    def grid_search_opt(model, params):
        '''
        Performs a grid search optimization on the given model/classifier
        using the given parameters.

        Returns results as DataFrame
        '''
        search = GridSearchCV(
            estimator=model, param_grid=params, scoring='accuracy', cv=3
        )
        with parallel_backend('threading'):
            search.fit(x, y)

        reg_results = pd.DataFrame(search.cv_results_)
        reg_results = reg_results.sort_values(by=['rank_test_score'])
        return reg_results

    params_svc = {'C': [0.1, 1, 10],
                  'degree': [2, 3, 4, 5],
                  'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}
    # Be mindful that the linear kernel takes a VERY long time to compute.

    params_rfc = {'n_estimators': [10, 50, 100],
                  'min_samples_split': [1.0, 2, 5]}  # Function requires 1.0 to be a float.

    reg_results = grid_search_opt(svc_model, params_svc)
    print(reg_results)

    reg_results = grid_search_opt(rfc_model, params_rfc)
    print(reg_results)


# --------------------------
# Final test on test dataset
# --------------------------
RUN_FINAL_TEST = True
if RUN_FINAL_TEST:
    [predictions, pred_accuracies, pred_metrics] = fit_classifier(x, x_test, y, y_test)

    print('Prediction accuracies (test set):')
    for prediction in pred_accuracies:
        print(f'{prediction}: {pred_accuracies[prediction]}')

    print('Prediction metrics (test set):')
    for prediction in pred_metrics:
        print(f'{prediction}: {pred_metrics[prediction]}')

    plot_confusion_matrix(svc_model, x_test, y_test)
    plt.show()
    plot_confusion_matrix(rfc_model, x_test, y_test)
    plt.show()
