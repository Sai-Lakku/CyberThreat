import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import category_encoders as ce
import ipaddress
from sklearn.feature_selection import RFE

# Load dataset
data = pd.read_csv('cybersecurity_attacks.csv')
data = data.dropna()
df = data.copy()

# Encode IP addresses (Source, Destination)
df['Source IP Encoded'] = df['Source IP Address'].apply(lambda ip: int(ipaddress.ip_address(ip)))
df['Destination IP Encoded'] = df['Destination IP Address'].apply(lambda ip: int(ipaddress.ip_address(ip)))

# Droping unnecessary features
drop_cols = ['Timestamp', 'Payload Data', 'User Information', 'Device Information',
             'Geo-location Data', 'Proxy Information', 'Source IP Address', 'Destination IP Address']
df.drop(columns=drop_cols, inplace=True)

# Binary encoding
binary_features = ['Packet Type', 'Malware Indicators', 'Alerts/Warnings', 'Firewall Logs', 'IDS/IPS Alerts', 'Log Source']
df = ce.BinaryEncoder(cols=binary_features).fit_transform(df)

# Ordinal encoding
df = ce.OrdinalEncoder(cols=['Severity Level']).fit_transform(df)

# Label encoding
df['Attack Type'] = LabelEncoder().fit_transform(df['Attack Type'])

# One-Hot Encoding
df = pd.get_dummies(df, columns=['Protocol', 'Traffic Type', 'Attack Signature', 'Action Taken', 'Network Segment'], dtype=int)

# Assign target and other features
X = df.drop(columns=['Attack Type'])
y = df['Attack Type']
original_columns = X.columns
X = pd.DataFrame(X, columns=original_columns)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Selection (RFE)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
rfe = RFE(estimator=rf, n_features_to_select=10)  # Select top 10 features
x_train_selected = rfe.fit_transform(x_train, y_train)
x_test_selected = rfe.transform(x_test)
selected_features = x_train.columns[rfe.support_]

# Convert train and test data to dataframe
x_train_selected = pd.DataFrame(x_train_selected, columns=selected_features)
x_test_selected = pd.DataFrame(x_test_selected, columns=selected_features)
#print(f"x_train columns: {x_train.columns}")
#print(f"x_test columns: {x_test.columns}")
print(x_train_selected.columns)

# Standardization
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_selected)
x_test_scaled = scaler.transform(x_test_selected)

# Convert scaled data to dataframes
x_train_scaled = pd.DataFrame(x_train_scaled, columns=selected_features)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=selected_features)

# Using SMOTE
smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train_scaled, y_train)
x_train_smote = pd.DataFrame(x_train_smote, columns=selected_features)
print(f"x_train_smote columns: {x_train_smote.columns}") # Print the columns after SMOTE

# Model configurations
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'LightGBM': LGBMClassifier(random_state=42),
    'MLP': MLPClassifier(random_state=42)
}

# Hyperparameter tuning
param_grid = {
    'RandomForest': {'n_estimators': [100, 200], 'max_depth': [10, 20], 'class_weight': ['balanced']},
    'XGBoost': {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]},
    'LightGBM': {'n_estimators': [100, 200], 'num_leaves': [31, 50], 'learning_rate': [0.05, 0.1]},
    'MLP': {'hidden_layer_sizes': [(64, 32), (128, 64)], 'max_iter': [1000, 1100], 'activation': ['relu']}
}

best_models = {}
for name, model in models.items():
    print(f"Tuning {name}...")
    search = RandomizedSearchCV(model, param_grid[name], n_iter=4, cv=3, scoring='accuracy', n_jobs=-1, verbose= 0, random_state=42)
    search.fit(x_train_smote, y_train_smote)
    best_models[name] = search.best_estimator_
    y_pred = best_models[name].predict(x_test_scaled)
    y_prob = best_models[name].predict_proba(x_test_scaled)
    print(f"{name} - ROC-AUC: {roc_auc_score(y_test, y_prob, multi_class='ovr'):.2f}")

# Stacking Classifier with XGBoost as final estimator
stacking_model = StackingClassifier(estimators=[('rf', best_models['RandomForest']), ('xgb', best_models['XGBoost']), ('lgbm', best_models['LightGBM']), ('mlp', best_models['MLP'])], final_estimator=XGBClassifier(random_state=42))
stacking_model.fit(x_train_smote, y_train_smote)

# Evaluate Stacking Classifier
y_prob_stacking = stacking_model.predict_proba(x_test_scaled)
print(f"Stacking - ROC-AUC: {roc_auc_score(y_test, y_prob_stacking, multi_class='ovr'):.2f}")
