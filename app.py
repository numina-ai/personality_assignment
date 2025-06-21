from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Extract and validate input
    try:
        features = [
            data['Time_spent_Alone'],
            int(data['Stage_fear']),
            data['Social_event_attendance'],
            data['Going_outside'],
            int(data['Drained_after_socializing']),
            data['Friends_circle_size'],
            data['Post_frequency']
        ]
    except KeyError as e:
        return jsonify({'error': f'Missing feature: {e}'}), 400

    # Scale and predict
    X_input = scaler.transform([features])
    prediction = model.predict(X_input)[0]

    return jsonify({
        'prediction': 'Extrovert' if prediction == 1 else 'Introvert'
    })

if __name__ == '__main__':
    app.run(debug=True)
