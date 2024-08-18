import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
from gpa_data_preprocessing import load_and_preprocess_data

# Load and preprocess data
data = load_and_preprocess_data()

# Select features and target
X = data[['gpa_like', 'sum_click', 'studied_credits', 'age_band', 'num_of_prev_attempts']]
y = data['final_result']

# Convert any remaining categorical variables if needed
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the trained model to a .pkl file
joblib.dump(model, 'gpa_predictor_model.pkl')
