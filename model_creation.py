import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

############### DATA IMPORT SECTION ########################################

# Dataset is not included as it is somewhat sensitive company internal data
# See ../Project_report.pdf for some insights into the data and more info about the project, results and conclusions
dataset = pd.read_csv('FINAL_DATASET.csv')  # See comments above

############ DATA PREPROCESSING SECTION ###################################
dataset = dataset.dropna()

# Debug: Print the first few rows of the merged DataFrame
print("Merged Data:")
print(dataset.head())

# Check if the merged DataFrame is empty
if dataset.empty:
    raise ValueError("Merged DataFrame is empty. Check the 'timestamp' values in the CSV files.")

# Prepare the features
features = ['warning_proportion', 'error_proportion', 'avg_cpu_usage_percent', 'avg_memory_usage_percent', 'avg_latency_milliseconds', 'avg_disk_usage_percent']
X = dataset[features]

# Check if the feature DataFrame is empty
if X.empty:
    raise ValueError("Feature DataFrame is empty. Ensure the features exist in the merged DataFrame.")

############ EXPLORATORY DATA ANALYSIS SECTION ###################################

# Calculate the correlation matrix
correlation_matrix = X.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.xticks(rotation=15, ha='right', fontsize=8)  # Adjust the fontsize as needed
plt.yticks(fontsize=8)  # Adjust the fontsize as needed
plt.show()

# Plot pair plot to show relationships between variables
sns.pairplot(X)
plt.suptitle('Pair Plot of Features', y=1.02)
plt.show()

################ PCA SECTION ####################################################

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA()
pca.fit(X_scaled)

# Calculate cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Plot explained variance ratio
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Explained Variance Ratio vs. Number of Components')
plt.grid(True)
plt.show()

# Select number of components (e.g., 90% explained variance)
n_components = np.argmax(cumulative_variance_ratio >= 0.9) + 1
print(f"Number of components explaining 90% variance: {n_components}")

# Transform data using selected number of components
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

print(f"X_pca top 5 rows: {X_pca[:5]}")
print(f"X top 5 rows: {X[:5]}")

############ ANOMALY DETECTION SECTION ####################################

# Create and fit the Isolation Forest model
contamination = 0.005  # Adjust this based on the expected anomaly rate
model = IsolationForest(contamination=contamination, random_state=42)
model.fit(X)

# Predict anomalies

# Start the timer
start_time = time.time()
# Perform the prediction
anomaly_labels = model.predict(X)
# End the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"Time taken for model.predict(X): {elapsed_time:.4f} seconds")
dataset['is_anomaly'] = anomaly_labels

# Function to detect anomalies in new data
def detect_anomalies(new_data):
    new_data_scaled = scaler.transform(new_data[features])
    # new_data_pca = pca.transform(new_data_scaled)
    predictions = model.predict(new_data_scaled)
    return predictions

print("-------------------------------------")
print("             RESULTS")
print("-------------------------------------")
print("Anomalies detected:")
# Drop the 'timestamp' column and print the anomalies
anomalies = dataset[dataset['is_anomaly'] == -1].drop(columns=['timestamp'])
print(anomalies)

# Example usage
# To use the model for new data:
# new_data = pd.DataFrame(...)  # New data with the same features
# new_anomalies = detect_anomalies(new_data)
