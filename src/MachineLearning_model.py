import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from Feature_selection import x_train_clean, y_train, x_test_clean, y_test
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=15, shuffle=True, random_state=42)

# Function to evaluate models using K-Fold CV
def evaluate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    return np.mean(scores)

# Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train_clean, y_train)
y_predictionRF = rf.predict(x_test_clean)
#accuracy_rf = accuracy_score(y_test, y_predictionRF)
accuracy_rf = evaluate_model(rf, x_train_clean, y_train)
#print(f'Accuracy of prediction using Random Forest model: {accuracy_rf:.2f}\n')

# XGBoost model
xgb = XGBClassifier(n_estimators=100, random_state=42)
xgb.fit(x_train_clean, y_train)
y_predictionXGB = xgb.predict(x_test_clean)
accuracy_xgb = accuracy_score(y_test, y_predictionXGB)
#accuracy_xgb = evaluate_model(xgb, x_train_clean, y_train)
print(f"Accuracy of prediction using XGBoost model: {accuracy_xgb:.2f}\n")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_clean, y_train)
y_predictionKNN = knn.predict(x_test_clean)
#accuracy_KNN = accuracy_score(y_test, y_predictionKNN)
accuracy_knn = evaluate_model(knn, x_train_clean, y_train)
print(f"Accuracy using KNN: {accuracy_knn:.2f}")





