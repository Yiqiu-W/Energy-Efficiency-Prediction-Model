from flask import Flask, request, render_template
import numpy as np
import joblib
import tensorflow as tf

app = Flask(__name__)

# Load TFLite models
heat_interpreter = tf.lite.Interpreter(model_path="best_heat_model.tflite")
heat_interpreter.allocate_tensors()

cool_interpreter = tf.lite.Interpreter(model_path="best_cool_model.tflite")
cool_interpreter.allocate_tensors()

# Load scaler
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

def run_tflite(interpreter, data):
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    interpreter.set_tensor(input_index, data)
    interpreter.invoke()
    return interpreter.get_tensor(output_index)[0][0]


@app.route("/")
def home():
    return render_template("index.html", feature_order=FEATURE_ORDER)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = []
        for f in FEATURE_ORDER:
            values.append(float(request.form[f]))

        X = np.array(values).reshape(1, -1)
        X_scaled = scaler.transform(X).astype(np.float32)

        heating = run_tflite(heat_interpreter, X_scaled)
        cooling = run_tflite(cool_interpreter, X_scaled)

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
