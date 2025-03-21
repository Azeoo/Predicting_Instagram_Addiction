from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model and scaler
model = joblib.load("logistic_model.pkl")  # Load logistic regression model
scaler = joblib.load("scaler.pkl")  # Load the trained scaler

# Custom threshold
THRESHOLD = 0.4

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

        # Return prediction result
        return jsonify({
            "addiction_status": addiction_status,
            "probability": round(p_prob, 4)  # Rounded probability for better readability
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
