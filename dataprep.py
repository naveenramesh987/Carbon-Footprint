import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
df = pd.read_csv('Carbon-Footprint/CarbonEmissions.csv')

# Split dataset into features and target
X = df.drop('CarbonEmission', axis=1)
y = df['CarbonEmission']

# Identify categorical columns
categorical_columns = ['Diet', 'Sex', 'Transport', 'Vehicle Type', 'Energy efficiency', 'Recycling']  # Add other categorical columns here

# Identify numerical columns
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Column Transformer to handle both numeric and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(), categorical_columns)
    ])

# Create a pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model_pipeline.fit(X_train, y_train)

# Test the model
print(f'Model Score: {model_pipeline.score(X_test, y_test)}')

