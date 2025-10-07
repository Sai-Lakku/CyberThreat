import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score

# Prepare dataset
data = pd.read_csv('Cybersecurity_Dataset.csv') # Import dataset
data = data.dropna() # Remove all null values

# Encode features into numerical values using Label Encoder
data['Threat Category'] = LabelEncoder().fit_transform(data['Threat Category'])
data['Threat Actor'] = LabelEncoder().fit_transform(data['Threat Actor'])
data['Attack Vector'] = LabelEncoder().fit_transform(data['Attack Vector'])
data['Geographical Location'] = LabelEncoder().fit_transform(data['Geographical Location'])
data['Suggested Defense Mechanism'] = LabelEncoder().fit_transform(data['Suggested Defense Mechanism'])
data['Cleaned Threat Description'] = LabelEncoder().fit_transform(data['Cleaned Threat Description'])
data['Keyword Extraction'] = LabelEncoder().fit_transform(data['Keyword Extraction'])
data['Named Entities (NER)'] = LabelEncoder().fit_transform(data['Named Entities (NER)'])
data['Topic Modeling Labels'] = LabelEncoder().fit_transform(data['Topic Modeling Labels'])
#print(data.head())

# Split data into test and train
features = ['Threat Actor','Attack Vector','Geographical Location','Named Entities (NER)','Keyword Extraction','Cleaned Threat Description','Suggested Defense Mechanism','Topic Modeling Labels','Severity Score','Sentiment in Forums','Risk Level Prediction'] 
goal = 'Threat Category' 
data[features] =  StandardScaler().fit_transform(data[features])
x = data[features] 
y = data[goal]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 

# Feature analysis
correlation = x.corrwith(data[goal]) # Pandas correalation
feature_correlation = pd.DataFrame(correlation, columns=["Correlation"])
rf = RandomForestClassifier(n_estimators=100, random_state=42) # Train the Random Forest model
rf.fit(x_train, y_train)
accuracy_before = rf.score(x_test, y_test) # Accuracy of the Random Forest model before feature selection
feature_importance = pd.DataFrame(rf.feature_importances_, index=x_train.columns, columns=['importance']).sort_values('importance', ascending=False) # Feature importance
print(f'\nAccuracy using Random Forest model before feature selection: {accuracy_before:.2f}')

# Remove and/or retain features
remove_features = ['Threat Actor','Attack vector', 'Topic Modeling Labels','Suggested Defense Mechanism']
retain_features = ['Sentiment in Forums','Risk Level Prediction','Keyword Extraction','Geographical Location','Named Entities (NER)','Cleaned Threat Description','Severity Score']
x_train_clean = x_train[retain_features].copy()
x_test_clean = x_test[retain_features].copy()
x_train_clean.loc[:, retain_features] = x_train_clean[retain_features].astype(float)
x_test_clean.loc[:, retain_features] = x_test_clean[retain_features].astype(float)

# Calculate final weights
feature_correlation["Correlation"] = np.abs(feature_correlation["Correlation"])
feature_correlation["Correlation"] /= feature_correlation["Correlation"].sum()
feature_importance["importance"] /= feature_importance["importance"].sum() # Normalize feature importance (ensure it sums to 1)
feature_weights = feature_importance.merge(feature_correlation, left_index=True, right_index=True) # Merge Importance and Correlation into a single DataFrame
alpha = 0.8  # Adjustable weight factor
feature_weights["FinalWeight"] = (alpha * feature_weights["importance"]) + ((1 - alpha) * feature_weights["Correlation"])
feature_weights["FinalWeight"] /= feature_weights["FinalWeight"].sum() # Normalize Final Weights
print("\nFeature Weights Based on Importance and Correlation:")
print(feature_weights.sort_values("FinalWeight", ascending=False))

# Apply weights to features
for feature, weight in feature_weights["FinalWeight"].items(): 
    if feature in retain_features:  # Ensure we only apply to the features to retain
        if feature in x_train_clean.columns:  # Check if the feature exists in x_train_clean
            x_train_clean[feature] = x_train_clean[feature] * weight
        if feature in x_test_clean.columns:  # Check if the feature exists in x_test_clean
            x_test_clean[feature] = x_test_clean[feature] * weight

# SMOTE 
smote = SMOTE(sampling_strategy='auto', random_state=32)
x_train_res, y_train_res = smote.fit_resample(x_train_clean, y_train)

# Hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [30, 50, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# RFM model accuracy
rf.fit(x_train_clean, y_train)
accuracy_after = rf.score(x_test_clean, y_test) # Accuracy 
print(f'\nAccuracy using Random Forest model after feature selection : {accuracy_after:.2f}\n')

# K-Fold cv function
kf = KFold(n_splits=15, shuffle=True, random_state=42)
def evaluate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    return np.mean(scores)

# XGBoost model
xgb = XGBClassifier(n_estimators=100, random_state=42)
xgb.fit(x_train_clean, y_train)
y_predictionXGB = xgb.predict(x_test_clean)
accuracy_xgb = accuracy_score(y_test, y_predictionXGB)
print(f"\nAccuracy of prediction using XGBoost model: {accuracy_xgb:.2f}\n")

# KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_clean, y_train)
y_predictionKNN = knn.predict(x_test_clean)
accuracy_knn = evaluate_model(knn, x_train_clean, y_train)
print(f"Accuracy using KNN: {accuracy_knn:.2f}\n")

# F-1 score
y_pred = rf.predict(x_test_clean)
f1 = f1_score(y_test, y_pred, average='macro')# F1 score
print(f"F1-Score: {f1:.2f}")

# Calculate ROC-AUC score
y_pred_proba = rf.predict_proba(x_test_clean) # Predict probabilities for each class
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')  # Use 'ovr' for multi-class
print(f"\nROC-AUC Score: {roc_auc:.2f}\n")


