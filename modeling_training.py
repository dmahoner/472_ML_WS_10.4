# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import math
import pickle

# Load the dataset
file_name = 'coffee_recommendation_dataset.xlsx'  # Ensure this file is in the same directory as your script
df = pd.read_excel(file_name)

# Data Preprocessing

# Encode categorical variables
label_encoder = LabelEncoder()

# Apply label encoding to the target column (Label)
df['Label'] = label_encoder.fit_transform(df['Label'])

# Save the label encoder for future use
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Apply one-hot encoding to the feature columns (Token_0 to Token_9)
df_encoded = pd.get_dummies(df, columns=[f'Token_{i}' for i in range(10)], drop_first=True)

# Splitting the dataset into training and test sets
X = df_encoded.drop('Label', axis=1)
y = df_encoded['Label']

# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data (for certain models like SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Machine Learning Models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=500),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(random_state=42)
}

# Training and evaluating the models
best_model = None
best_rmse = float('inf')

for model_name, model in models.items():
    print(f"Training {model_name}...")

    # Use scaled data for SVM, for others, use non-scaled data
    if model_name == "SVM":
        model.fit(X_train_scaled, y_train)
        y_test_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)

    # Calculate RMSE for the test dataset
    rmse_test = math.sqrt(mean_squared_error(y_test, y_test_pred))

    # Save the best model based on RMSE for the test dataset
    if rmse_test < best_rmse:
        best_rmse = rmse_test
        best_model = model
        best_model_name = model_name

    print(f"{model_name} RMSE on Test Data: {rmse_test}")

# Save the best model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"Best model: {best_model_name} with RMSE: {best_rmse}")

import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model
with open('best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as label_encoder_file:
    label_encoder = pickle.load(label_encoder_file)

# Title of the app
st.title("Coffee Type Prediction")

# Sidebar inputs for user preferences
st.sidebar.header("User Preferences")

time_of_day = st.sidebar.selectbox("Time of Day", ['morning', 'afternoon', 'evening'])
coffee_strength = st.sidebar.selectbox("Coffee Strength", ['mild', 'regular', 'strong'])
sweetness_level = st.sidebar.selectbox("Sweetness Level", ['unsweetened', 'lightly sweetened', 'sweet'])
milk_type = st.sidebar.selectbox("Milk Type", ['none', 'regular', 'skim', 'almond'])
coffee_temperature = st.sidebar.selectbox("Coffee Temperature", ['hot', 'iced', 'cold brew'])
flavored_coffee = st.sidebar.selectbox("Flavored Coffee", ['yes', 'no'])
caffeine_tolerance = st.sidebar.selectbox("Caffeine Tolerance", ['low', 'medium', 'high'])
coffee_bean = st.sidebar.selectbox("Coffee Bean", ['Arabica', 'Robusta', 'blend'])
coffee_size = st.sidebar.selectbox("Coffee Size", ['small', 'medium', 'large'])
dietary_preferences = st.sidebar.selectbox("Dietary Preferences", ['none', 'vegan', 'lactose-intolerant'])

# Encoding the inputs manually (same encoding as in your training data)
input_data = pd.DataFrame({
    'Token_0': [time_of_day],
    'Token_1': [coffee_strength],
    'Token_2': [sweetness_level],
    'Token_3': [milk_type],
    'Token_4': [coffee_temperature],
    'Token_5': [flavored_coffee],
    'Token_6': [caffeine_tolerance],
    'Token_7': [coffee_bean],
    'Token_8': [coffee_size],
    'Token_9': [dietary_preferences]
})

# One-hot encode the input data (ensure it matches the training data)
input_encoded = pd.get_dummies(input_data)

# Align columns with the training data (required columns)
required_columns = model.feature_names_in_  # Get the feature columns from the model
for col in required_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[required_columns]

# Make the prediction
prediction = model.predict(input_encoded)[0]

# Reverse the label encoding (map the prediction back to the coffee type)
coffee_type = label_encoder.inverse_transform([prediction])[0]

# Display the prediction
st.subheader(f"Recommended Coffee: {coffee_type}")