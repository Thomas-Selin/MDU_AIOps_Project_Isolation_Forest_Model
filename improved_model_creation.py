import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self, contamination=0.005, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.features = [
            'warning_proportion', 'error_proportion', 'avg_cpu_usage_percent',
            'avg_memory_usage_percent', 'avg_latency_milliseconds', 'avg_disk_usage_percent'
        ]
        self.scaler = StandardScaler()
        self.model = None
        
    def load_and_preprocess_data(self, filepath):
        """Load and preprocess the dataset."""
        try:
            dataset = pd.read_csv(filepath)
            logger.info(f"Loaded dataset with shape: {dataset.shape}")
            
            # Data validation
            missing_cols = set(self.features) - set(dataset.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Handle missing values
            initial_rows = len(dataset)
            dataset = dataset.dropna(subset=self.features)
            dropped_rows = initial_rows - len(dataset)
            if dropped_rows > 0:
                logger.warning(f"Dropped {dropped_rows} rows with missing values")
            
            return dataset
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def perform_eda(self, X):
        """Perform exploratory data analysis."""
        output_dir = Path("improved_model_creation_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.xticks(rotation=15, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_matrix.png')
        plt.close()
        
        # Distribution plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, feature in enumerate(self.features):
            sns.histplot(data=X, x=feature, ax=axes[idx])
            axes[idx].set_title(f'{feature} Distribution')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_distributions.png')
        plt.close()
        
        # Box plots
        plt.figure(figsize=(12, 6))
        X.boxplot(column=self.features)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_boxplots.png')
        plt.close()
        
        # Perform PCA analysis (for dimensionality analysis only)
        X_scaled = self.scaler.fit_transform(X)
        pca = PCA()
        pca.fit(X_scaled)
        
        # Plot explained variance ratio
        plt.figure(figsize=(10, 5))
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('PCA Analysis: Explained Variance Ratio')
        plt.grid(True)
        plt.savefig(output_dir / 'pca_analysis.png')
        plt.close()
        
        # Log PCA results
        logger.info("\nPCA Analysis Results:")
        for i, ratio in enumerate(pca.explained_variance_ratio_):
            logger.info(f"Component {i+1}: {ratio:.3f} of variance explained")
        logger.info(f"Number of components for 90% variance: {np.argmax(cumulative_variance >= 0.9) + 1}")
        
        # Save component loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(len(self.features))],
            index=self.features
        )
        loadings.to_csv(output_dir / 'pca_loadings.csv')
        
        logger.info("EDA plots and PCA analysis saved to 'improved_model_creation_outputs' directory")

    def train_model(self, dataset):
        """Train the anomaly detection model using raw (scaled) features."""
        try:
            X = dataset[self.features]
            
            # Split data into train/test sets
            X_train, X_test = train_test_split(X, test_size=0.2, random_state=self.random_state)
            
            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train Isolation Forest
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1  # Use all available cores
            )
            
            # Train and measure time
            start_time = time.time()
            self.model.fit(X_train_scaled)
            training_time = time.time() - start_time
            
            logger.info(f"Model training completed in {training_time:.2f} seconds")
            
            # Evaluate on test set
            test_predictions = self.model.predict(X_test_scaled)
            anomaly_rate = (test_predictions == -1).mean()
            logger.info(f"Anomaly rate on test set: {anomaly_rate:.2%}")
            
            # Feature importance analysis
            feature_importance = self.analyze_feature_importance(X_test)
            for feature, importance in feature_importance.items():
                logger.info(f"Feature importance - {feature}: {importance:.3f}")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

    def analyze_feature_importance(self, X):
        """Analyze feature importance by measuring impact on anomaly scores."""
        feature_importance = {}
        X_scaled = self.scaler.transform(X)
        base_score = np.abs(self.model.score_samples(X_scaled)).mean()
        
        for i, feature in enumerate(self.features):
            X_modified = X_scaled.copy()
            X_modified[:, i] = 0  # Zero out one feature
            modified_score = np.abs(self.model.score_samples(X_modified)).mean()
            feature_importance[feature] = abs(modified_score - base_score)
            
        # Normalize importance scores
        max_importance = max(feature_importance.values())
        feature_importance = {k: v/max_importance for k, v in feature_importance.items()}
        
        return feature_importance

    def detect_anomalies(self, new_data):
        """Detect anomalies in new data."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
            
        try:
            # Validate input features
            missing_features = set(self.features) - set(new_data.columns)
            if missing_features:
                raise ValueError(f"Missing features in input data: {missing_features}")
            
            # Transform data
            new_data_scaled = self.scaler.transform(new_data[self.features])
            
            # Predict
            start_time = time.time()
            predictions = self.model.predict(new_data_scaled)
            anomaly_scores = self.model.score_samples(new_data_scaled)
            prediction_time = time.time() - start_time
            
            logger.info(f"Predictions completed in {prediction_time:.4f} seconds")
            
            # Add predictions and scores to dataframe
            result_df = new_data.copy()
            result_df['is_anomaly'] = predictions
            result_df['anomaly_score'] = anomaly_scores
            
            # Calculate anomaly statistics
            anomaly_rate = (predictions == -1).mean()
            logger.info(f"Detected anomaly rate: {anomaly_rate:.2%}")
            
            # Analyze anomalies
            if len(result_df[result_df['is_anomaly'] == -1]) > 0:
                self._analyze_anomalies(result_df[result_df['is_anomaly'] == -1])
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            raise

    def _analyze_anomalies(self, anomalies_df):
        """Analyze detected anomalies."""
        logger.info("\nAnomaly Analysis:")
        logger.info(f"Total anomalies detected: {len(anomalies_df)}")
        
        # Feature statistics for anomalies
        for feature in self.features:
            mean_val = anomalies_df[feature].mean()
            std_val = anomalies_df[feature].std()
            logger.info(f"{feature}:")
            logger.info(f"  Mean: {mean_val:.2f}")
            logger.info(f"  Std: {std_val:.2f}")

def main():
    # Initialize detector
    detector = AnomalyDetector(contamination=0.005)
    
    try:
        # Load and process data
        # Dataset is not included as consists of somewhatwhat sensitive company internal data
        # See the folder improved_model_creation_outputs for more insights into the data and results
        # Also, the Project_report.pdf contains a detailed explanation of the project
        dataset = detector.load_and_preprocess_data('FINAL_DATASET.csv')
        
        # Perform EDA (including PCA analysis)
        detector.perform_eda(dataset[detector.features])
        
        # Train model (using raw features)
        X_train, X_test = detector.train_model(dataset)
        
        # Detect anomalies
        results = detector.detect_anomalies(dataset)
        
        # Save results
        results.to_csv('anomaly_detection_results.csv', index=False)
        anomalies = results[results['is_anomaly'] == -1]
        anomalies.to_csv('detected_anomalies.csv', index=False)
        
        logger.info(f"Saved {len(anomalies)} anomalies to 'detected_anomalies.csv'")
        logger.info(f"Full results saved to 'anomaly_detection_results.csv'")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
