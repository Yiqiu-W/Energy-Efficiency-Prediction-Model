from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import joblib
import numpy as np

app = Flask(__name__)

# -------- LOAD MODELS + SCALER --------
heat_model = tf.keras.models.load_model("best_heat_model.keras")
cool_model = tf.keras.models.load_model("best_cool_model.keras")
scaler = joblib.load("scaler.pkl")

FEATURE_ORDER = [
    "Relative_Compactness",
    "Surface_Area",
    "Wall_Area",
    "Roof_Area",
    "Overall_Height",
    "Orientation",
    "Glazing_Area",
    "Glazing_Area_Distribution"
]

# -------- HOME PAGE --------
@app.route("/")
def home():
    return render_template("index.html", feature_order=FEATURE_ORDER)

# -------- PREDICT (POST from the form) --------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = []
        for f in FEATURE_ORDER:
            values.append(float(request.form[f]))

        X = np.array(values).reshape(1, -1)
        X_scaled = scaler.transform(X)

        heating = float(heat_model.predict(X_scaled)[0][0])
        cooling = float(cool_model.predict(X_scaled)[0][0])

        return render_template(
            "index.html",
            feature_order=FEATURE_ORDER,
            heating=heating,
            cooling=cooling,
            filled_values=request.form
        )

    except Exception as e:
        return render_template("index.html", feature_order=FEATURE_ORDER, error=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
