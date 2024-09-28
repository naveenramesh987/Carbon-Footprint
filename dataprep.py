from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('Carbon-Footprint/carbon_footprint_model.joblib')

# Define the column names (headers) for the input features
feature_columns = ['Body Type', 'Sex', 'Diet', 'How Often Shower', 'Heating Energy Source', 'Transport', 
                   'Vehicle Type', 'Social Activity', 'Monthly Grocery Bill', 'Frequency of Traveling by Air', 
                   'Vehicle Monthly Distance Km', 'Waste Bag Size', 'Waste Bag Weekly Count', 
                   'How Long TV PC Daily Hour', 'How Many New Clothes Monthly', 'How Long Internet Daily Hour', 
                   'Energy efficiency', 'Recycling', 'Cooking_With']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from the form
    features = [
        request.form['body_type'],
        request.form['sex'],
        request.form['diet'],
        request.form['how_often_shower'],
        request.form['heating_energy_source'],
        request.form['transport'],
        request.form['vehicle_type'],
        request.form['social_activity'],
        request.form['monthly_grocery_bill'],
        request.form['frequency_travel_air'],
        request.form['vehicle_distance_km'],
        request.form['waste_bag_size'],
        request.form['waste_bag_count'],
        request.form['tv_pc_hours'],
        request.form['new_clothes_count'],
        request.form['internet_hours'],
        request.form['energy_efficiency'],
        request.form['recycling'],
        request.form['cooking_with']
    ]

    # Convert the features into a pandas DataFrame with appropriate column names
    features_df = pd.DataFrame([features], columns=feature_columns)

    # Make prediction
    prediction = model.predict(features_df)

    return render_template('result.html', carbon_emission=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
