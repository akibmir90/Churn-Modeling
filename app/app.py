from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the best model
best_model = joblib.load('models/best_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        # Extract data from the form
    }

    # Preprocess data (similar to what you did in data_preprocessing.py)
    # ...

    # Make prediction using the best model
    churn_prediction = best_model.predict([list(data.values())])

    return f"Churn Prediction: {churn_prediction[0]}"

if __name__ == '__main__':
    app.run(debug=True)
