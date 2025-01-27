# Credit-Card-Approval-Prediction-Using-Machine-Learning
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create a sample dataset
data = {
    "Age": np.random.randint(20, 70, 500),
    "Income": np.random.randint(20000, 100000, 500),
    "Credit_Score": np.random.randint(300, 850, 500),
    "Employment_Status": np.random.choice([0, 1, 2], 500),  # 0 = Unemployed, 1 = Employed, 2 = Self-Employed
    "Loan_Amount": np.random.randint(5000, 50000, 500),
    "Debt": np.random.randint(1000, 20000, 500),
    "Approval_Status": np.random.choice([0, 1], 500)  # 0 = Rejected, 1 = Approved
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save dataset
df.to_csv("credit_card_approval.csv", index=False)
print("Dataset saved successfully!")

# Load dataset
df = pd.read_csv("credit_card_approval.csv")

# Display first 5 rows
print(df.head())

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Fill missing values with the median
df.fillna(df.median(), inplace=True)

# Define features (X) and target variable (y)
X = df.drop("Approval_Status", axis=1)
y = df["Approval_Status"]

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print model accuracy
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "credit_card_approval_model.pkl")

# Save scaler for future predictions
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved successfully!")

# Load trained model and scaler
model = joblib.load("credit_card_approval_model.pkl")
scaler = joblib.load("scaler.pkl")

# Collect user inputs
print("\nEnter the following details to check credit card approval:")
age = int(input("Enter Age: "))
income = float(input("Enter Monthly Income ($): "))
credit_score = float(input("Enter Credit Score (300-850): "))
employment_status = int(input("Employment Status (0 = Unemployed, 1 = Employed, 2 = Self-Employed): "))
loan_amount = float(input("Enter Loan Amount ($): "))
debt = float(input("Enter Current Debt ($): "))

# Convert user input into an array
user_data = np.array([[age, income, credit_score, employment_status, loan_amount, debt]])

# Scale input using the saved scaler
user_data_scaled = scaler.transform(user_data)

# Predict approval status
prediction = model.predict(user_data_scaled)

# Display result
if prediction[0] == 1:
    print("\nCongratulations! Your credit card application is Approved.")
else:
    print("\nSorry, your credit card application is Rejected.")
