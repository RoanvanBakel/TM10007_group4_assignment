from ecg.load_data import load_data

data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')


import pandas as pd

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

def perform_pca(train_df, validation_df):
    # Perform PCA
    pca = PCA(n_components = .90)
    principal_components_train = pca.fit_transform(train_df)
    principal_components_validation = pca.transform(validation_df)

    df_pc_train = pd.DataFrame(data=principal_components_train)
    df_pc_validation = pd.DataFrame(data=principal_components_validation)

    return df_pc_train, df_pc_validation

x, x_test = perform_pca(x, x_test) 

assert all(x) <= 1

# -----------------------
# K-fold Cross-validation
# -----------------------
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits = 10, shuffle = True)
for train_index, test_index in skf.split(x, y):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # TODO Make sure classifier is run for each fold.

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

SVC_model.fit(x_train, y_train)
KNN_model.fit(x_train, y_train)
LG_model.fit(x_train, y_train)
DTR_model.fit(x_train, y_train)
RFC_model.fit(x_train, y_train)
GNB_model.fit(x_train, y_train)


SVC_prediction = SVC_model.predict(x_test)
KNN_prediction = KNN_model.predict(x_test)
LG_prediction = LG_model.predict(x_test)
DTR_prediction = DTR_model.predict(x_test)
RFC_prediction = DTR_model.predict(x_test)
GNB_prediction = DTR_model.predict(x_test)

# Accuracy score is the simplest way to evaluate
print(accuracy_score(SVC_prediction, y_test))
print(accuracy_score(KNN_prediction, y_test))
print(accuracy_score(LG_prediction, y_test))
print(accuracy_score(DTR_prediction, y_test))
print(accuracy_score(RFC_prediction, y_test))
print(accuracy_score(GNB_prediction, y_test))
# But Confusion Matrix and Classification Report give more details about performance
print(classification_report(KNN_prediction, y_test))

print(x)