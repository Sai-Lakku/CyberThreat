import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from Feature_selection import data, goal
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

# Further feature engineering
data['Risk_Severity_Interaction'] = data['Risk Level Prediction'] * data['Severity Score']
data['Risk_Sentiment_Interaction'] = data['Risk Level Prediction'] * data['Sentiment in Forums']
data['Risk_Sentiment_Severity_Interaction'] = data['Risk_Sentiment_Interaction'] * data['Risk_Severity_Interaction']

# Scaling 
numerical_values = ['Risk Level Prediction', 'Severity Score', 'Sentiment in Forums','Risk_Sentiment_Interaction', 'Risk_Severity_Interaction']
data[numerical_values] =  StandardScaler().fit_transform(data[numerical_values])

poly = PolynomialFeatures(interaction_only=True, include_bias=False)
interaction_features = poly.fit_transform(data[['Risk_Sentiment_Interaction', 'Sentiment in Forums']])
# Add polynomial interaction features to the dataset
interaction_data = pd.DataFrame(interaction_features, columns=['Int_Feature1', 'Int_Feature2', 'Int_Feature3'])
data = pd.concat([data.reset_index(drop=True), interaction_data], axis=1)


# Remove and/or retain features
remove_features = ['Threat Actor','Attack vector', 'Topic Modeling Labels','Suggested Defense Mechanism']
retain_features = ['Risk_Sentiment_Severity_Interaction','Keyword Extraction','Cleaned Threat Description','Geographical Location','Named Entities (NER)']

# Split data
x_new = data[retain_features] 
y_new = data[goal]
x_new_train, x_new_test, y_new_train, y_new_test = train_test_split(x_new, y_new, test_size=0.2, random_state=42) 

# Feature analysis
correlation = x_new.corrwith(data[goal]) # Pandas correalation
print("Feature enginnering correaltion:\n", correlation)
rf = RandomForestClassifier(n_estimators=100, random_state=42) # Train the Random Forest model
rf.fit(x_new_train, y_new_train)
accuracy_before = rf.score(x_new_test, y_new_test) # Accuracy of the Random Forest model before feature selection
feature_importances = pd.DataFrame(rf.feature_importances_, index=x_new_train.columns, columns=['importance']).sort_values('importance', ascending=False) # Feature importance
print(feature_importances) # Feature importance sorted in descending order
print(f'\nAccuracy using Random Forest model after feature enginnering: {accuracy_before:.2f}\n')

import seaborn as sns
import matplotlib.pyplot as plt

# Compute the correlation matrix
correlation_matrix = data.corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))

# Generate a heatmap of the correlation matrix
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, cbar=True)

# Show the plot
plt.title('Feature Correlation Heatmap')
plt.show()