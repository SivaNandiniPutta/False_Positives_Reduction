#!/usr/bin/env python3
"""
SecureFilter AI - Flask Backend
Trains a Random Forest model using CICIDS2017 cleaned dataset
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
import joblib
from model import EnhancedSecurityAlertML

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
CORS(app)

# Paths
DATA_PATH = 'cicids2017_cleaned.csv'
MODELS_FOLDER = 'models'
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Global model instance
ml_model = EnhancedSecurityAlertML()

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        # Check if dataset exists
        if not os.path.exists(DATA_PATH):
            return jsonify({'error': f'Dataset not found at {DATA_PATH}'}), 404

        # Load the dataset
        df = pd.read_csv(DATA_PATH)
        logger.info(f"✅ Dataset loaded from {DATA_PATH}, shape: {df.shape}")

        # Check if label column exists (adjust based on your dataset)
        label_col = 'Label'  # Common in CICIDS2017, adjust if different
        if label_col not in df.columns:
            # Try alternative column names
            possible_labels = ['is_false_positive', 'label', 'class', 'attack_type']
            for col in possible_labels:
                if col in df.columns:
                    label_col = col
                    break
            else:
                return jsonify({'error': f'No suitable label column found. Available columns: {list(df.columns)}'}), 400

        # Train the model
        acc, roc = ml_model.train_model(df, label_col=label_col)

        # Save the model
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        model_path = os.path.join(MODELS_FOLDER, f"model_{timestamp}.joblib")
        ml_model.save_model(model_path)
        logger.info(f"💾 Model saved to {model_path}")

        return jsonify({
            'message': 'Model trained successfully',
            'accuracy': float(acc),
            'roc_auc': float(roc),
            'model_path': model_path
        })

    except Exception as e:
        logger.error(f"❌ Error during training: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/performance', methods=['GET'])
def get_performance():
    """Get model performance statistics"""
    try:
        # Mock data - replace with actual performance tracking
        performance_data = {
            'before': {
                'false_positives': 850,
                'accuracy': 75.2
            },
            'after': {
                'false_positives': 120,
                'accuracy': 94.8
            }
        }
        return jsonify(performance_data)
    except Exception as e:
        logger.error(f"❌ Error getting performance: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/alerts', methods=['GET'])
def get_alerts():
    """Get filtered security alerts"""
    try:
        # Generate mock alert data - replace with actual data processing
        mock_alerts = []
        alert_types = ['Port Scan', 'DDoS Attack', 'Malware', 'Brute Force', 'SQL Injection']
        predictions = ['True Positive', 'False Positive']
        
        for i in range(10):
            alert = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'src_ip': f'192.168.1.{100 + i}',
                'dst_ip': f'10.0.0.{50 + i}',
                'alert': np.random.choice(alert_types),
                'prediction': np.random.choice(predictions, p=[0.8, 0.2])  # 80% true positive
            }
            mock_alerts.append(alert)
        
        return jsonify({'alerts': mock_alerts})
    except Exception as e:
        logger.error(f"❌ Error getting alerts: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions on new data"""
    try:
        data = request.get_json()
        if not ml_model.model:
            return jsonify({'error': 'Model not trained yet. Please train the model first.'}), 400
        
        # Convert input data to DataFrame
        df = pd.DataFrame([data])
        prediction = ml_model.predict(df)
        
        return jsonify({
            'prediction': prediction[0],
            'confidence': float(np.max(ml_model.model.predict_proba(df)))
        })
    except Exception as e:
        logger.error(f"❌ Error making prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_trained': ml_model.model is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
