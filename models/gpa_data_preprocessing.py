import pandas as pd
import numpy as np


def load_and_preprocess_data():
    # Load the datasets
    student_info = pd.read_csv('../datasets/studentInfo.csv')
    assessments = pd.read_csv('../datasets/assessments.csv')
    student_assessment = pd.read_csv('../datasets/studentAssessment.csv')
    student_vle = pd.read_csv('../datasets/studentVle.csv')

    # Merge datasets
    student_performance = pd.merge(student_info, student_assessment, on='id_student')
    student_performance = pd.merge(student_performance, assessments, on='id_assessment')

    # Example: Create a feature for the total number of interactions with VLE
    student_vle_grouped = student_vle.groupby('id_student')['sum_click'].sum().reset_index()
    student_performance = pd.merge(student_performance, student_vle_grouped, on='id_student', how='left')

    # Calculate GPA-like measure: sum of weighted scores divided by sum of credits
    student_performance['weighted_score'] = student_performance['score'] * student_performance['weight']
    gpa_like_measure = student_performance.groupby('id_student').apply(
        lambda x: x['weighted_score'].sum() / x['weight'].sum() if x['weight'].sum() != 0 else np.nan
    ).reset_index(name='gpa_like')

    # Merge the GPA-like measure back to the main dataset
    student_performance = pd.merge(student_performance, gpa_like_measure, on='id_student')

    # Handle missing values
    student_performance.dropna(inplace=True)

    # Encode categorical variables (like age_band)
    age_band_mapping = {
        '0-35': 1,
        '35-55': 2,
        '55<=': 3
    }
    student_performance['age_band'] = student_performance['age_band'].map(age_band_mapping)

    # Map final_result to binary outcome (Pass = 1, Fail = 0)
    student_performance['final_result'] = student_performance['final_result'].map({
        'Pass': 1, 'Fail': 0, 'Withdrawn': 0, 'Distinction': 1
    })

    return student_performance
