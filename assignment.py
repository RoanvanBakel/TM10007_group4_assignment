from ecg.load_data import load_data

data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')

# --------------
# Data splitting
# --------------
from sklearn.model_selection import train_test_split 

labels = data.pop('label')

x, x_test, y, y_test = train_test_split (data, labels, test_size=0.2, train_size=0.8, stratify=True)

# ---------------
# Feature scaling
# ---------------
from sklearn.preprocessing import RobustScaler, QuantileTransformer
scaler = RobustScaler() 
scaler.fit_transform(x)


from sklearn.decomposition import PCA

def perform_pca(train_df, validation_df):
    
    # Perform PCA
    pca = PCA(n_components = .90)
    principal_components_train = pca.fit_transform(train_df)
    principal_components_validation = pca.transform(validation_df)

    df_pc_train = pd.DataFrame(data=principal_components_train)
    df_pc_validation = pd.DataFrame(data=principal_components_validation)

    return df_pc_train, df_pc_validation

pc_train, pc_validation = perform_pca(x_train_rescaled, x_test_rescaled) 

print(pc_test)

from sklearn.model_selection import StratifiedKFold
y = data['label']
X = data.loc[:, data.columns != 'label']
skf = StratifiedKFold(n_splits = 10, shuffle = True)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

[15:35, 25/03/2021] +31 6 14617133: # Begin by importing all necessary libraries
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
KNN_model = KNeighborsClassifier(n_neighbors=5)
LG_model = LogisticRegression(max_iter=10000)
DTR_model = DecisionTreeClassifier()
RFC_model = RandomForestClassifier()
GNB_model = GaussianNB()

SVC_model.fit(X_train, y_train)
KNN_model.fit(X_train, y_train)
LG_model.fit(X_train, y_train)
DTR_model.fit(X_train, y_train)
RFC_model.fit(X_train, y_train)
GNB_model.fit(X_train, y_train)


SVC_prediction = SVC_model.predict(X_test)
KNN_prediction = KNN_model.predict(X_test)
LG_prediction = LG_model.predict(X_test)
DTR_prediction = DTR_model.predict(X_test)
RFC_prediction = DTR_model.predict(X_test)
GNB_prediction = DTR_model.predict(X_test)

# Accuracy score is the simplest way to evaluate

print(accuracy_score(SVC_prediction, y_test))
print(accuracy_score(KNN_prediction, y_test))
print(accuracy_score(LG_prediction, y_test))
print(accuracy_score(DTR_prediction, y_test))
print(accuracy_score(RFC_prediction, y_test))
print(accuracy_score(GNB_prediction, y_test))
# But Confusion Matrix and Classification Report give more details about performance
print(classification_report(KNN_prediction, y_test))
print(classification_report(SVC_prediction, y_test))

print(X)