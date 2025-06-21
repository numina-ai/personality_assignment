from flask import Flask, request, jsonify 
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model (make sure 'svm_model.pkl' is in the same directory)
model = pickle.load(open('svm_model.pkl', 'rb'))

# Simple encoding for categorical features
def preprocess(data):
    return [
        data["Time_spent_Alone"],
        1 if data["Stage_fear"] == "Yes" else 0,
        data["Social_event_attendance"],
        data["Going_outside"],
        1 if data["Drained_after_socializing"] == "Yes" else 0,
        data["Friends_circle_size"],
        data["Post_frequency"]
    ]

@app.route("/")
def home():
    return "Personality Predictor API is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = preprocess(data)
        prediction = model.predict([features])[0]
        result = "Introvert" if prediction == "Introvert" else "Extrovert"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
