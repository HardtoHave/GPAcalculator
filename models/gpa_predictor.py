import numpy as np
import joblib
import pandas as pd

model = joblib.load('models/gpa_predictor_model.pkl')


def calculate_current_gpa(courses):
    total_weighted_marks = 0
    total_credit_points = 0

    for course in courses:
        mark = course['mark']
        credit_points = course['credit_points']

        # Skip courses with mark or credit points equal to 0
        if mark > 0 and credit_points > 0:
            total_weighted_marks += mark * credit_points
            total_credit_points += credit_points

    # Avoid division by zero
    if total_credit_points == 0:
        return 0.0

    gpa = total_weighted_marks / total_credit_points
    return round(gpa, 2)


def predict_gpa(current_gpa, courses, student_info_features):
    # Prepare the feature vector in the same structure as the training data
    features_dict = {
        'gpa_like': current_gpa,
        'sum_click': student_info_features['sum_click'],
        'studied_credits': student_info_features['studied_credits'],
        'age_band': student_info_features['age_band'],
        'num_of_prev_attempts': student_info_features['num_of_prev_attempts']
    }

    # Convert the dictionary to a DataFrame with the correct order of columns
    features_df = pd.DataFrame([features_dict])

    # Ensure that the feature columns are in the same order as the training set
    features_df = features_df[['gpa_like', 'sum_click', 'studied_credits', 'age_band', 'num_of_prev_attempts']]

    # Predict the next semester's GPA using the model
    predicted_gpa = model.predict(features_df)[0]

    return current_gpa + round(predicted_gpa, 2)
