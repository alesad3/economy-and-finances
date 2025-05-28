"""
Credit Card Fraud Detection with Multiple ML Methods
===================================================
This script implements several machine learning methods for credit card fraud detection:
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Logistic Regression
- Autoencoder Neural Network

Features:
- Data preprocessing and scaling
- Handling class imbalance with SMOTE
- Comprehensive model evaluation metrics
- Model comparison and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_curve, auc, roc_curve, precision_score,
    recall_score, f1_score, roc_auc_score
)
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
import time
import os
import joblib
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Try to prevent TensorFlow from using GPU if not needed
try:
    tf.config.set_visible_devices([], "GPU")  # Hide GPUs from TensorFlow
except:
    pass


class CreditCardFraudDetection:
    """
    A class for credit card fraud detection using various machine learning methods.
    """

    def __init__(self, data_path="creditcard.csv"):
        """
        Initialize the fraud detection class.

        Args:
            data_path (str): Path to the credit card dataset
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_resampled = None
        self.y_train_resampled = None
        self.scaler = StandardScaler()
        self.results = {}
        self.models = {}

        # Create directories for saving results
        os.makedirs("models", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs("plots", exist_ok=True)

    def load_data(self):
        """Load the credit card dataset and display basic information."""
        print("Loading data from", self.data_path)
        self.df = pd.read_csv(self.data_path)

        # Display basic information
        print("\nDataset Shape:", self.df.shape)
        print("\nClass Distribution:")
        class_counts = self.df["Class"].value_counts()
        print(class_counts)
        print(f"Fraud percentage: {class_counts[1] / len(self.df) * 100:.4f}%")

        # Check for missing values
        missing_values = self.df.isnull().sum().sum()
        print(f"\nMissing values: {missing_values}")

        return self.df

    def explore_data(self):
        """Perform exploratory data analysis and generate visualizations."""
        print("\nPerforming exploratory data analysis...")

        # Plot class distribution
        plt.figure(figsize=(8, 6))
        sns.countplot(x='Class', data=self.df)
        plt.title('Class Distribution (0: Normal, 1: Fraud)')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.savefig('plots/class_distribution.png')

        # Plot correlation matrix for a subset of features
        plt.figure(figsize=(12, 10))
        subset_df = self.df[['V1', 'V2', 'V3', 'V4', 'V5', 'Amount', 'Class']]
        correlation_matrix = subset_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix of Selected Features')
        plt.tight_layout()
        plt.savefig('plots/correlation_matrix.png')
        plt.close('all')

        print("Plots saved to 'plots' directory.")
        return self.df

    def preprocess_data(self, test_size=0.2):
        """
        Preprocess the data, including splitting, scaling, and handling imbalance.

        Args:
            test_size (float): Proportion of the dataset to include in the test split
        """
        print("\nPreprocessing data...")

        # Separate features and target
        X = self.df.drop("Class", axis=1)
        y = self.df["Class"]

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")

        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        # Apply SMOTE to handle class imbalance
        print("\nApplying SMOTE to handle class imbalance...")
        smote = SMOTE(random_state=42)
        self.X_train_resampled, self.y_train_resampled = smote.fit_resample(self.X_train, self.y_train)

        # Check class distribution after SMOTE
        print("\nClass distribution after SMOTE:")
        print(pd.Series(self.y_train_resampled).value_counts())

        print("Preprocessing complete.")

    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Evaluate a trained model and store the results.

        Args:
            model: The trained model
            X_test: Test features
            y_test: Test labels
            model_name (str): Name of the model for reporting
        """
        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_pred_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_prob = model.predict(X_test)

        y_pred = (y_pred_prob > 0.5).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Compute ROC AUC if possible
        try:
            roc_auc = roc_auc_score(y_test, y_pred_prob)
        except:
            roc_auc = None

        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': confusion,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_pred_prob
        }

        # Print results
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        if roc_auc is not None:
            print(f"ROC AUC: {roc_auc:.4f}")

        print("\nConfusion Matrix:")
        print(confusion)

        print("\nClassification Report:")
        print(report)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(f'plots/{model_name.lower().replace(" ", "_")}_confusion_matrix.png')

        # Plot ROC curve if possible
        if roc_auc is not None:
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
            plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.4f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{model_name} ROC Curve')
            plt.legend(loc='lower right')
            plt.savefig(f'plots/{model_name.lower().replace(" ", "_")}_roc_curve.png')

        plt.close('all')

    def train_knn(self):
        """Train a K-Nearest Neighbors classifier."""
        print("\n" + "=" * 50)
        print("Training K-Nearest Neighbors Classifier")
        print("=" * 50)

        start_time = time.time()

        # Initialize and train KNN
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(self.X_train_resampled, self.y_train_resampled)

        # Evaluate
        self.evaluate_model(knn, self.X_test, self.y_test, "KNN")

        # Save model
        joblib.dump(knn, 'models/knn_model.pkl')

        end_time = time.time()
        print(f"Training and evaluation completed in {end_time - start_time:.2f} seconds")

        self.models['KNN'] = knn
        return knn

    def train_naive_bayes(self):
        """Train a Gaussian Naive Bayes classifier."""
        print("\n" + "=" * 50)
        print("Training Gaussian Naive Bayes Classifier")
        print("=" * 50)

        start_time = time.time()

        # Initialize and train Naive Bayes
        gnb = GaussianNB()
        gnb.fit(self.X_train_resampled, self.y_train_resampled)

        # Evaluate
        self.evaluate_model(gnb, self.X_test, self.y_test, "Naive Bayes")

        # Save model
        joblib.dump(gnb, 'models/naive_bayes_model.pkl')

        end_time = time.time()
        print(f"Training and evaluation completed in {end_time - start_time:.2f} seconds")

        self.models['Naive Bayes'] = gnb
        return gnb

    def train_logistic_regression(self):
        """Train a Logistic Regression classifier with class weights."""
        print("\n" + "=" * 50)
        print("Training Logistic Regression Classifier")
        print("=" * 50)

        start_time = time.time()

        # Compute class weights
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=np.unique(self.y_train), y=self.y_train
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        print("Class weights:", class_weight_dict)

        # Initialize and train Logistic Regression
        lr = LogisticRegression(
            solver="lbfgs",
            random_state=42,
            class_weight=class_weight_dict,
            max_iter=200
        )
        lr.fit(self.X_train_resampled, self.y_train_resampled)

        # Evaluate
        self.evaluate_model(lr, self.X_test, self.y_test, "Logistic Regression")

        # Feature importance
        feature_names = self.df.drop("Class", axis=1).columns
        coefficients = lr.coef_[0]

        # Create a DataFrame of feature importances
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.abs(coefficients)
        }).sort_values('Importance', ascending=False)

        print("\nTop 10 most important features:")
        print(feature_importance.head(10))

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
        plt.title('Logistic Regression Feature Importance')
        plt.tight_layout()
        plt.savefig('plots/logistic_regression_feature_importance.png')
        plt.close()

        # Save model
        joblib.dump(lr, 'models/logistic_regression_model.pkl')

        end_time = time.time()
        print(f"Training and evaluation completed in {end_time - start_time:.2f} seconds")

        self.models['Logistic Regression'] = lr
        return lr

    def train_autoencoder(self, epochs=30, batch_size=32):
        """
        Train an autoencoder for anomaly detection.

        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
        """
        print("\n" + "=" * 50)
        print("Training Autoencoder for Anomaly Detection")
        print("=" * 50)

        start_time = time.time()

        # Define model architecture
        input_dim = self.X_train.shape[1]
        encoding_dim = 14  # Same as in the original script

        # Define the input layer
        input_layer = Input(shape=(input_dim,))

        # Define the encoder layers
        encoder = Dense(28, activation="relu")(input_layer)
        encoder = Dense(encoding_dim, activation="relu")(encoder)

        # Define the decoder layers
        decoder = Dense(28, activation="relu")(encoder)
        decoder = Dense(input_dim, activation="linear")(decoder)

        # Define the autoencoder model
        autoencoder = Model(inputs=input_layer, outputs=decoder)

        # Compile the model
        autoencoder.compile(optimizer="adam", loss="mse")

        # Add early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # Train the model
        print("\nTraining autoencoder...")
        history = autoencoder.fit(
            self.X_train_resampled,
            self.X_train_resampled,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=1
        )

        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Autoencoder Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('plots/autoencoder_training_loss.png')
        plt.close()

        # Use autoencoder for anomaly detection
        print("\nUsing autoencoder for anomaly detection...")

        # Reconstruct test data
        reconstructed_X_test = autoencoder.predict(self.X_test)

        # Calculate MSE for each sample
        mse = np.mean(np.power(self.X_test - reconstructed_X_test, 2), axis=1)

        # Determine threshold (95th percentile of reconstruction errors)
        threshold = np.percentile(mse, 95)
        print(f"Anomaly threshold: {threshold:.6f}")

        # Predict anomalies (1 if MSE > threshold, 0 otherwise)
        y_pred = (mse > threshold).astype(int)

        # Calculate evaluation metrics
        print("\nAutoencoder Results:")
        accuracy = accuracy_score(self.y_test, y_pred)
        confusion = confusion_matrix(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nConfusion Matrix:\n{confusion}")
        print(f"\nClassification Report:\n{report}")

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
        plt.title('Autoencoder Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig('plots/autoencoder_confusion_matrix.png')

        # Plot reconstruction error distribution
        plt.figure(figsize=(10, 6))
        plt.hist(mse[self.y_test == 0], bins=50, alpha=0.5, label='Normal')
        plt.hist(mse[self.y_test == 1], bins=50, alpha=0.5, label='Fraud')
        plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.6f}')
        plt.title('Reconstruction Error Distribution')
        plt.xlabel('Reconstruction Error (MSE)')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig('plots/autoencoder_reconstruction_error.png')
        plt.close()

        # Store results
        self.results['Autoencoder'] = {
            'accuracy': accuracy,
            'confusion_matrix': confusion,
            'classification_report': report,
            'predictions': y_pred,
            'mse': mse,
            'threshold': threshold,
            'model': autoencoder
        }

        # Save model
        autoencoder.save('models/autoencoder_model.h5')

        end_time = time.time()
        print(f"Training and evaluation completed in {end_time - start_time:.2f} seconds")

        self.models['Autoencoder'] = autoencoder
        return autoencoder

    def compare_models(self):
        """Compare all trained models and visualize the results."""
        print("\n" + "=" * 50)
        print("Model Comparison")
        print("=" * 50)

        if not self.results:
            print("No models have been trained yet.")
            return

        # Prepare data for comparison
        models = []
        accuracy = []
        precision = []
        recall = []
        f1 = []
        roc_auc = []

        for model_name, result in self.results.items():
            models.append(model_name)
            accuracy.append(result.get('accuracy', 0))

            if model_name != 'Autoencoder':
                precision.append(result.get('precision', 0))
                recall.append(result.get('recall', 0))
                f1.append(result.get('f1_score', 0))
                roc_auc.append(result.get('roc_auc', 0))
            else:
                # For autoencoder, calculate precision, recall and F1 from confusion matrix
                cm = result.get('confusion_matrix', np.zeros((2, 2)))
                tn, fp, fn, tp = cm.ravel()
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_score = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

                precision.append(prec)
                recall.append(rec)
                f1.append(f1_score)
                roc_auc.append(0)  # Autoencoder doesn't provide probability scores

        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Model': models,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC AUC': roc_auc
        })

        # Sort by F1 score
        comparison_df = comparison_df.sort_values('F1 Score', ascending=False)

        print("\nModel Performance Comparison:")
        print(comparison_df)

        # Save comparison to CSV
        comparison_df.to_csv('results/model_comparison.csv', index=False)

        # Plot comparison metrics
        plt.figure(figsize=(12, 8))

        # Bar width
        width = 0.15

        # Positions
        r1 = np.arange(len(models))
        r2 = [x + width for x in r1]
        r3 = [x + width for x in r2]
        r4 = [x + width for x in r3]
        r5 = [x + width for x in r4]

        # Create bars
        plt.bar(r1, comparison_df['Accuracy'], width=width, label='Accuracy')
        plt.bar(r2, comparison_df['Precision'], width=width, label='Precision')
        plt.bar(r3, comparison_df['Recall'], width=width, label='Recall')
        plt.bar(r4, comparison_df['F1 Score'], width=width, label='F1 Score')
        plt.bar(r5, comparison_df['ROC AUC'], width=width, label='ROC AUC')

        # Add labels and title
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks([r + width * 2 for r in range(len(models))], comparison_df['Model'])
        plt.legend()

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('plots/model_comparison.png')
        plt.close()

        print("\nModel comparison completed. Results saved to 'results/model_comparison.csv'")
        print("Comparison plot saved to 'plots/model_comparison.png'")

        return comparison_df


def main():
    """Main function to run the credit card fraud detection pipeline."""
    data_path = "creditcard.csv"

    fraud_detector = CreditCardFraudDetection(data_path)

    fraud_detector.load_data()
    fraud_detector.explore_data()
    fraud_detector.preprocess_data()

    # Train models
    print("\nTraining models...")

    # Train KNN classifier
    fraud_detector.train_knn()

    # Train Naive Bayes classifier
    fraud_detector.train_naive_bayes()

    # Train Logistic Regression classifier
    fraud_detector.train_logistic_regression()

    # Train Autoencoder
    fraud_detector.train_autoencoder()

    # Compare models
    comparison = fraud_detector.compare_models()

    print("\nFraud detection pipeline completed successfully!")
    print("Models saved to 'models' directory")
    print("Results saved to 'results' directory")
    print("Plots saved to 'plots' directory")


if __name__ == "__main__":
    main()
