# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load your dataset
df = pd.read_csv(r'C:\Users\prabh\Desktop\co2\newCo2.csv')

# Replace empty values in 'Vehicle Type' with 'Not Specified'
df['Vehicle Type'].replace('', 'Not Specified', inplace=True)

# Define features (X) and target (y)
X = df.drop('CarbonEmission', axis=1)
y = df['CarbonEmission']

# Identify categorical and numerical columns
categorical_columns = ['Body Type', 'Sex', 'Diet', 'How Often Shower', 'Heating Energy Source',
                       'Transport', 'Vehicle Type', 'Social Activity', 
                       'Frequency of Traveling by Air', 'Waste Bag Size', 
                       'How Long Internet Daily Hour', 'Energy efficiency', 
                       'Recycling', 'Cooking_With', 'Combined']

numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Create a Column Transformer to handle both numeric and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(drop='first'), categorical_columns)
    ])

# Create a pipeline with preprocessing and the RandomForestRegressor model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model_pipeline.fit(X_train, y_train)

# Evaluate the model
model_score = model_pipeline.score(X_test, y_test)
print(f'Model Score: {model_score}')

# Save the model to a file
# joblib.dump(model_pipeline, 'carbon_emission_model_pipeline.joblib')

print("Model training complete and saved as 'carbon_emission_model_pipeline.joblib'")



from flask import Flask, render_template, request
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('carbon_emission_model.joblib')

# Define route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Define route to handle prediction logic
@app.route('/predict', methods=['POST'])
def predict():
    # Extract form inputs
    body_type = request.form['body_type']
    sex = request.form['sex']
    diet = request.form['diet']
    how_often_shower = request.form['how_often_shower']
    heating_energy_source = request.form['heating_energy_source']
    combined = request.form.get("combined", '[]')
    social_activity = request.form['social_activity']
    grocery_bill = float(request.form['monthly_grocery_bill'])
    air_travel = request.form['frequency_travel_air']
    vehicle_distance = float(request.form['vehicle_distance_km'])
    waste_bag_size = request.form['waste_bag_size']
    waste_bag_count = int(request.form['waste_bag_count'])
    tv_pc_hours = float(request.form['tv_pc_hours'])
    clothes_count = int(request.form['new_clothes_count'])
    internet_hours = float(request.form['internet_hours'])
    energy_efficiency = request.form['energy_efficiency']
    recycling = request.form.get('recycling', '[]')
    cooking_with = request.form.get('cooking_with', '[]')

    # Create a DataFrame from the inputs
    input_data = pd.DataFrame({
        'Body Type': [body_type],
        'Sex': [sex],
        'Diet': [diet],
        'How Often Shower': [how_often_shower],
        'Heating Energy Source': [heating_energy_source],
        'Transport Type': [combined],
        'Social Activity': [social_activity],
        'Monthly Grocery Bill': [grocery_bill],
        'Frequency of Traveling by Air': [air_travel],
        'Vehicle Monthly Distance Km': [vehicle_distance],
        'Waste Bag Size': [waste_bag_size],
        'Waste Bag Weekly Count': [waste_bag_count],
        'How Long TV PC Daily Hour': [tv_pc_hours],
        'How Many New Clothes Monthly': [clothes_count],
        'How Long Internet Daily Hour': [internet_hours],
        'Energy efficiency': [energy_efficiency],
        'Recycling': [recycling],
        'Cooking_With': [cooking_with]
    })

    # Perform one-hot encoding of categorical features
    input_encoded = pd.get_dummies(input_data)

    # Align input features with model features (fill missing columns with 0)
    model_features = model.feature_names_in_
    input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

    # Predict the carbon emissions
    prediction = model.predict(input_encoded)

    # Return the result to the user
    return render_template('result.html', carbon_emission=prediction[0])


if __name__ == '__main__':
    app.run(debug=True)
