import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "heart_disease_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:

        age = float(request.form["age"])
        sex = int(request.form["sex"])
        cp = int(request.form["cp"])
        trestbps = float(request.form["trestbps"])
        chol = float(request.form["chol"])
        fbs = int(request.form["fbs"])
        restecg = int(request.form["restecg"])
        thalach = float(request.form["thalach"])
        exang = int(request.form["exang"])
        oldpeak = float(request.form["oldpeak"])
        slope = int(request.form["slope"])
        ca = int(request.form["ca"])
        thal = int(request.form["thal"])

        data = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }])

        # Scale
        data_scaled = scaler.transform(data)

        # Prediction
        prediction = model.predict(data_scaled)[0]

        probability = model.predict_proba(data_scaled)[0]

        disease_probability = probability[1] * 100
        healthy_probability = probability[0] * 100

        if prediction == 1:
            result = "❤️ Heart Disease Detected"
            color = "red"
        else:
            result = "💚 No Heart Disease"
            color = "green"

        return render_template(
            "index.html",
            prediction_text=result,
            disease_probability=round(disease_probability, 2),
            healthy_probability=round(healthy_probability, 2),
            color=color
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}",
            color="red"
        )

if __name__ == "__main__":
    app.run(debug=True)