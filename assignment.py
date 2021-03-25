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