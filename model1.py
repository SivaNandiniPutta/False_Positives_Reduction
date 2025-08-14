import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import logging

logger = logging.getLogger(__name__)

class EnhancedSecurityAlertML:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_binary_classification = True

    def preprocess_data(self, df: pd.DataFrame, label_col: str, is_training: bool = True):
        """Preprocess the data for training or prediction"""
        df_processed = df.copy()
        
        # Separate features and target
        if label_col in df_processed.columns:
            y = df_processed[label_col]
            X = df_processed.drop(columns=[label_col])
        else:
            X = df_processed
            y = None
        
        # Handle missing values
        X = X.fillna(0)
        
        # Convert categorical columns to numeric
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X[col] = pd.Categorical(X[col]).codes
        
        # Convert all columns to numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        if is_training:
            # Store feature columns for later use
            self.feature_columns = X.columns.tolist()
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            # Encode labels if needed
            if y is not None:
                if y.dtype == 'object':
                    y = self.label_encoder.fit_transform(y)
                    self.is_binary_classification = len(np.unique(y)) == 2
                else:
                    # Convert to binary if needed (assume 0 = benign, 1 = malicious)
                    unique_values = y.unique()
                    if len(unique_values) > 2:
                        # Multi-class to binary: assume first class is benign
                        y = (y != unique_values[0]).astype(int)
                    self.is_binary_classification = True
        else:
            # Ensure same columns as training
            if self.feature_columns:
                # Add missing columns with zeros
                for col in self.feature_columns:
                    if col not in X.columns:
                        X[col] = 0
                # Remove extra columns and reorder
                X = X[self.feature_columns]
            
            # Scale features using fitted scaler
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X, y

    def train_model(self, df: pd.DataFrame, label_col: str = 'Label'):
        """Train the Random Forest model"""
        try:
            if label_col not in df.columns:
                raise ValueError(f"Label column '{label_col}' not found. Available columns: {list(df.columns)}")
            
            logger.info(f"Training model with dataset shape: {df.shape}")
            
            # Preprocess data
            X, y = self.preprocess_data(df, label_col, is_training=True)
            
            # Check for empty dataset
            if X.empty or y is None:
                raise ValueError("Dataset is empty or has no valid features")
            
            logger.info(f"Preprocessed data shape: X={X.shape}, y={len(y)}")
            logger.info(f"Class distribution: {np.bincount(y)}")
            
            # Train/validation split
            X_train, X_valid, y_train, y_valid = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Train model with balanced class weights
            self.model = RandomForestClassifier(
                n_estimators=100,  # Reduced for faster training
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            logger.info("Training Random Forest model...")
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_valid)
            acc = accuracy_score(y_valid, y_pred)
            
            # Calculate ROC AUC
            if self.is_binary_classification:
                y_prob = self.model.predict_proba(X_valid)[:, 1]
                roc = roc_auc_score(y_valid, y_prob)
            else:
                roc = 0.0  # Placeholder for multi-class
            
            logger.info(f"Model training completed. Accuracy: {acc:.4f}, ROC-AUC: {roc:.4f}")
            logger.info(f"Classification Report:\n{classification_report(y_valid, y_pred)}")
            
            return acc, roc
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Preprocess data
        X_processed, _ = self.preprocess_data(X, label_col=None, is_training=False)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        # Convert back to original labels if needed
        if hasattr(self.label_encoder, 'classes_'):
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions

    def predict_proba(self, X: pd.DataFrame):
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Preprocess data
        X_processed, _ = self.preprocess_data(X, label_col=None, is_training=False)
        
        # Get probabilities
        probabilities = self.model.predict_proba(X_processed)
        return probabilities

    def save_model(self, path: str):
        """Save the trained model and preprocessing components"""
        if self.model is None:
            raise ValueError("No model to save. Please train the model first.")
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'is_binary_classification': self.is_binary_classification
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load a trained model"""
        try:
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.is_binary_classification = model_data['is_binary_classification']
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
