# False_Positives_Reduction
Intelligent IT Operations – Reducing False Positives in Security Alerts A machine learning-powered solution using the CICIDS2017 dataset and Random Forest to filter false positives in network alerts. Includes a Flask backend and interactive dashboard to boost SOC efficiency and ensure faster, accurate threat detection.
# Project Title
Intelligent IT Operations – Reducing False Positives in Security Alerts

# Overview
This project is a full-stack application designed to intelligently analyze security alerts and reduce false positives in large-scale IT environments. Using a machine learning model trained on real-world security alert data, the system classifies alerts and provides actionable insights through a web-based dashboard.

# Project Structure
- `model.py` – Trains and evaluates the ML model using network security data.
- `app.py` – Flask backend with APIs for predictions and data upload.
- `templates/dashboard.html` – Frontend dashboard for interacting with the system.

#  Code Explanations

## dashboard.html
This is the main user interface built with HTML and JavaScript. It uses Bootstrap for layout and Chart.js for visualization.

Key Features:
- **Upload File Button** – Sends a CSV file to the Flask backend via `POST`.
- **Submit Button** – Requests predictions from the backend.
- **Visualizations** – Displays:
  - Attack type distribution
  - Prediction outcome pie chart
  - Bar chart of actual vs predicted labels

## model.py
This script handles ML model training.

Workflow:
- Loads and cleans a large CSV dataset of network traffic and alerts.
- Uses `RandomForestClassifier` to learn from labeled data.
- Performs preprocessing: drops NaNs, encodes labels, splits into train/test.
- Reports classification performance: accuracy, precision, recall, F1-score.
- Saves the trained model to `model.pkl`.

## app.py
Backend server built with Flask.

Endpoints:
- `/` – Serves the `dashboard.html` page.
- `/predict` – Accepts a CSV file, loads it, and returns predicted labels.
- `/analyze` – Analyzes predictions and returns aggregated statistics for frontend visualizations.

Other Features:
- Uses CORS for cross-origin access.
- Loads the model with `joblib`.
- Supports real-time prediction and dashboard refresh.

# Technologies Used
- Python 3.x
- Flask
- Scikit-learn
- Pandas, NumPy
- HTML, CSS, JavaScript
- Bootstrap
- Chart.js

#  Dataset
Due to size limitations, the dataset is **not included in this repository**. You may download the dataset (e.g., CICIDS2017) and place it in the same directory before running `model.py`.

Expected format:
- CSV file with numerical features.
- Must contain a target column named `Attack Type` (or modify accordingly).

- Link of Dataset : https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed

#  How to Run

##  Step 1: Install Dependencies

pip install flask scikit-learn pandas numpy joblib flask-cors


##  Step 2: Train the Model
python model.py


##  Step 3: Run the Flask App
python app.py

##  Step 4: Open Dashboard
Open http://127.0.0.1:5000/ in your browser. Upload a CSV and click "Submit" to view the results.

python model.py
