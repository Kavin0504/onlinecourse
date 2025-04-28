import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = "D:/KAVINKUMAR/online_course_engagement_data.csv"  # Update your actual file path
df = pd.read_csv(file_path)

# Check for missing values and drop them
df = df.dropna()

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Convert categorical variables using Label Encoding
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Encode categorical values
    label_encoders[col] = le  # Store the encoder for future use

# Define target variable (Assuming last column is the target)
target_column = df.columns[-1]  # Change this if the target column is different
X = df.drop(columns=[target_column])  # Features
y = df[target_column]  # Target variable

# Select top 5 features based on correlation with target variable
correlations = X.corrwith(y).abs().sort_values(ascending=False)
top_5_features = correlations.index[:5]
X_selected = X[top_5_features]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions for evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Streamlit App
st.title("Online Course Engagement Predictor")
st.write(f"### Model Performance\n**Mean Squared Error:** {mse:.2f}")

st.write("## Enter feature values to predict engagement:")

input_data = {}

for feature in top_5_features:
    value = st.number_input(f"Enter value for {feature}", value=0.0)
    input_data[feature] = [value]

if st.button("Predict"):
    input_df = pd.DataFrame(input_data)
    prediction = model.predict(input_df)
    st.success(f"Predicted {target_column}: {prediction[0]:.2f}")
