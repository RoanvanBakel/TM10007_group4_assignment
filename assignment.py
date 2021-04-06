from ecg.load_data import load_data

data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')


import pandas as pd
import numpy as np

# --------------
# Data splitting
# --------------
from sklearn.model_selection import train_test_split 

labels = data.pop('label')

x, x_test, y, y_test = train_test_split(data, labels, test_size=0.2, train_size=0.8, stratify=labels) # TODO should we stratify on more than the labels alone?

# ---------------
# Feature scaling
# ---------------
from sklearn.preprocessing import RobustScaler, QuantileTransformer
scaler = RobustScaler() 
scaler.fit_transform(x)


# ----------------------------------
# Principal Component Analysis (PCA)
# ----------------------------------
from sklearn.decomposition import PCA

def perform_pca(train_df, validation_df): # TODO hoor je dit niet alleen op de trainingsset te doen? Maar hoe?
    # Perform PCA
    pca = PCA(n_components = .90)
    principal_components_train = pca.fit_transform(train_df)
    principal_components_validation = pca.transform(validation_df)

    df_pc_train = pd.DataFrame(data=principal_components_train)
    df_pc_validation = pd.DataFrame(data=principal_components_validation)

    return df_pc_train, df_pc_validation

x, x_test = perform_pca(x, x_test) 

assert all(x) <= 1

# ----------
# Classifier
# ----------
# Begin by importing all necessary libraries
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

SVC_model = SVC()
KNN_model = KNeighborsClassifier(n_neighbors=10)
LG_model = LogisticRegression(max_iter=10000)
DTR_model = DecisionTreeClassifier()
RFC_model = RandomForestClassifier()
GNB_model = GaussianNB()

def fit_classifier(x_train, x_test, y_train, y_test):
    SVC_model.fit(x_train, y_train)
    KNN_model.fit(x_train, y_train)
    LG_model.fit(x_train, y_train)
    DTR_model.fit(x_train, y_train)
    RFC_model.fit(x_train, y_train)
    GNB_model.fit(x_train, y_train)

    predictions = {}
    predictions['SVC_prediction'] = SVC_model.predict(x_test)
    predictions['KNN_prediction'] = KNN_model.predict(x_test)
    predictions['LG_prediction'] = LG_model.predict(x_test)
    predictions['DTR_prediction'] = DTR_model.predict(x_test)
    predictions['RFC_prediction'] = DTR_model.predict(x_test)
    predictions['GNB_prediction'] = DTR_model.predict(x_test)

    pred_accuracies = {}
    for pred in predictions:
        pred_accuracies[pred] = accuracy_score(predictions[pred], y_test)

    pred_metrics = {}
    for pred in predictions:
        pred_metrics[pred] = classification_report(predictions[pred], y_test, zero_division=0)

    return predictions, pred_accuracies, pred_metrics


# -----------------------
# K-fold Cross-validation
# -----------------------
from sklearn.model_selection import StratifiedKFold

k = 10
skf = StratifiedKFold(n_splits = k, shuffle = True)
all_pred_accuracies = {}
for train_index, test_index in skf.split(x, y):
    [predictions, pred_accuracies, pred_metrics] = fit_classifier(x.iloc[train_index],
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
# Grid Search Regularization
# --------------------------
run_grid_search = True
if run_grid_search:
    def grid_search_reg(model, params):
        from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
        from sklearn.utils import parallel_backend

        search = GridSearchCV(
            estimator=model, param_grid=params, scoring='accuracy', cv=3
        )
        with parallel_backend('threading'):
            search.fit(x, y)

        reg_results = pd.DataFrame(search.cv_results_)
        reg_results = reg_results.sort_values(by=['rank_test_score'])
        return reg_results

    params_SVC = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf','linear','sigmoid']}
    params_RFC = {'n_estimators': [10,50,100],
                  'min_samples_split': [1,2,5]}
    reg_results = grid_search_reg(RFC_model, params_RFC)
    print(reg_results)

# --------------------------
# Final test on test dataset
# --------------------------

run_final_test = False
if run_final_test:
    [predictions, pred_accuracies, pred_metrics] = classifier(x_train, x_test, y_train, y_test)

    for prediction in pred_accuracies:
        print(f'{prediction}: {pred_accuracies[prediction]}')

    for prediction in pred_metrics:
        print(f'{prediction}: {pred_metrics[prediction]}')