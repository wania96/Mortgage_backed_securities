# app.py
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the saved pipelines for classification and regression
with open("classification_.pkl", 'rb') as file:
    classification_pipeline = pickle.load(file)

with open("regression_.pkl", 'rb') as file:
    regression_pipeline = pickle.load(file)

# Load the combined predictions DataFrame
with open("combined_predictions.pkl", 'rb') as file:
    combined_predictions = pickle.load(file)

# Function to get predictions based on user input
def get_predictions(prediction_type, data):
    if prediction_type == "classification":
        predictions = classification_pipeline.predict(data)
    else:
        predictions = regression_pipeline.predict(data)
    return predictions.flatten()

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the prediction type from the request data
        prediction_type = request.json.get('prediction_type')

        # Get the user input data from the request data
        input_data = request.json.get('input_data')

        # Create DataFrame from input data
        data = pd.DataFrame([input_data])

        # Perform predictions
        predictions = get_predictions(prediction_type, data)

        # Prepare response
        response = {
            "predictions": predictions.tolist()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
