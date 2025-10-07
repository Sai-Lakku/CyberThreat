import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from MachineLearning_model import y_predictionRF
from First_Dataset import x_test_clean, data


risk_data = pd.DataFrame({'Risk Level Prediction': x_test_clean['Risk Level Prediction'], 'Threat Category': y_predictionRF})
severity_data = pd.DataFrame({'Severity Score': x_test_clean['Severity Score'], 'Threat Category': y_predictionRF})

custom_labels = ['DDos', 'Malware', 'Phishing', 'Ransomware']

plt.figure(figsize=(10, 6))
sns.boxplot(x='Threat Category', y='Risk Level Prediction', data=risk_data)
plt.title("Risk Level Distribution by Threat Category")
plt.xlabel("Threat Category")
plt.ylabel("Risk Level Prediction")
plt.xticks(ticks=range(len(custom_labels)), labels=custom_labels)
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Threat Category', y='Severity Score', data=severity_data)
plt.title("Severity Score Distribution by Threat Category")
plt.xlabel("Threat Category")
plt.ylabel("Severity Score Prediction")
plt.xticks(ticks=range(len(custom_labels)), labels=custom_labels)
plt.show()