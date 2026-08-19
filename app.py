from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

import os

app = Flask(__name__)

# Load Logistic Regression model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "student_stress_logistic_model2.pkl")
model = joblib.load(MODEL_PATH)


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get form data from HTML
        data = request.form

        # Create student DataFrame
        student = pd.DataFrame([{
            "academic_pressure_score": float(data["academic_pressure_score"]),
            "anxiety_score": float(data["anxiety_score"]),
            "depression_score": float(data["depression_score"]),
            "social_support_score": float(data["social_support_score"]),
            "screen_time_hours": float(data["screen_time_hours"]),
            "daily_sleep_hours": float(data["daily_sleep_hours"]),
            "attendance_percentage": float(data["attendance_percentage"]),
            "cgpa": float(data["cgpa"])
        }])

        # Predict stress level
        prediction = model.predict(student)[0]

        # Get probabilities
        probabilities = model.predict_proba(student)[0]

        probability_data = []

        for class_name, probability in zip(
            model.classes_,
            probabilities
        ):
            probability_data.append({
                "class_name": class_name,
                "probability": round(probability * 100, 2)
            })

        # Render result in HTML
        return render_template("index.html", prediction=prediction, probabilities=probability_data)

    except Exception as e:
        return render_template("index.html", error=str(e))


# Run Flask
if __name__ == "__main__":
    app.run(debug=True)