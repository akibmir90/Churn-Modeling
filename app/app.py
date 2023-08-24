from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the best trained model
best_model = joblib.load('best_model.pkl')

# Location label mapping
location_mapping = {
    'Los Angeles': 0,
    'New York': 1,
    'Miami': 2,
    'Chicago': 3,
    'Houston': 4
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        age = int(request.form['age'])
        gender = request.form['gender']
        location = request.form['location']
        subscription_months = int(request.form['subscription_months'])
        monthly_bill = float(request.form['monthly_bill'])
        total_usage_gb = float(request.form['total_usage_gb'])
        
        # Prepare the input data for prediction
        gender_encoded = 1 if gender == 'Male' else 0
        location_encoded = location_mapping[location]
        
        input_features = [age, subscription_months, monthly_bill, total_usage_gb, gender_encoded, location_encoded]
        
        # Make a prediction
        prediction = best_model.predict([input_features])[0]
        
        # Convert prediction to human-readable format
        churn_label = 'Churn' if prediction == 1 else 'Not Churn'
        
        return render_template('index.html', prediction=churn_label)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
application = app
