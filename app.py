from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model and scaler
model = joblib.load("logistic_model.pkl")  # Load logistic regression model
level_model = joblib.load("level_addiction_model.pkl")  # Model for addiction level (0, 1, 2, or 3)
scaler = joblib.load("scaler.pkl")  # Load the trained scaler

# Custom threshold
THRESHOLD = 0.4

# Define feature names (must match training data)
FEATURE_COLUMNS = [
    "time_spent", "frequency_opened", "fomo_intensity", "disrupted_sleep",
    "difficulty_reducing_usage", "escape_mechanism", "interference_with_work",
    "late_night_usage", "anxious_without_use", "disrupted_routine"
]


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON request data
        data = request.get_json()
        # print("Received JSON:", data)

        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in request data"}), 400

        # Convert input data to numpy array and reshape
        features = np.array(data["features"]).reshape(1, -1)
        # print("Raw Features:", features)

        features_level = pd.DataFrame(features, columns=FEATURE_COLUMNS) # Resolving pandas error RandomForest model for column name

        # Fix: Convert to DataFrame with correct column names
        features_df = pd.DataFrame(features, columns=scaler.feature_names_in_)

        # Apply StandardScaler correctly
        features_scaled = scaler.transform(features_df)
        # print("Scaled Features:", features_scaled)

        # Get probability for the "Addicted" class (class 1)
        p_prob = model.predict_proba(features_scaled)[:, 1][0]
        # print("Addiction Probability:", p_prob)

        # Predict class using the custom threshold (0.4)
        addiction_status = int(p_prob > THRESHOLD)  # 1 = Addicted, 0 = Not Addicted
        # print("Predicted Addiction Status:", addiction_status)

        # 🔹 **Second Model: Predict Addiction Level (Use RAW Data)**
        addiction_level = level_model.predict(features_level)[0]  # Use original unscaled data


        # Return both predictions to frontend
        return jsonify({
            "addiction_status": addiction_status,  # 0 = Not Addicted, 1 = Addicted
            "probability": round(p_prob, 4),  # Probability of addiction
            "addiction_level": int(addiction_level)  # Level (0, 1, 2, 3)
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
