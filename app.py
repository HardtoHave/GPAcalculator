from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from models.gpa_predictor import predict_gpa, calculate_current_gpa
from models.upstage_api import extract_academic_transcript
import random

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'docx'}  # Allowed file types


# Function to check allowed file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Upload page route
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Assuming the API returns detected courses as part of the analysis
            detected_courses = extract_academic_transcript(file_path)

            return render_template('upload.html', detected_courses=detected_courses)
    return render_template('upload.html')


@app.route('/show', methods=['POST'])
def show_page():
    # Get the modified courses data from the form
    courses = []
    subject_names = request.form.getlist('subject_name[]')
    marks = request.form.getlist('mark[]')
    credit_points = request.form.getlist('credit_points[]')

    for subject_name, mark, credit_point in zip(subject_names, marks, credit_points):
        courses.append({
            'subject_name': subject_name,
            'mark': float(mark),
            'credit_points': float(credit_point)
        })

    # Calculate current GPA
    current_gpa = calculate_current_gpa(courses)

    # Predict next semester's GPA
    # For simplicity, let's assume we pass some dummy student info features for now
    student_info_features = {
        'sum_click': 100,  # Example feature
        'age_band': 2,  # Example feature
        'num_of_prev_attempts': 1,  # Example feature
        'studied_credits': sum([course['credit_points'] for course in courses if course['credit_points'] > 0])
    }
    predicted_gpa = predict_gpa(current_gpa, courses, student_info_features)
    if predicted_gpa > current_gpa:
        encouragements = [
            "You're improving! Keep pushing your limits!",
            "Great work! Your efforts are paying off!",
            "Fantastic! You're on an upward trajectory!",
            "You're getting better and better—keep it up!",
            "Your hard work is clearly showing results!"
        ]
    elif predicted_gpa < current_gpa:
        encouragements = [
            "Don't worry, setbacks happen—keep striving!",
            "It's okay, everyone has off days. Keep going!",
            "Remember, it's about the journey. You can bounce back!",
            "Stay positive! This is just a learning opportunity.",
            "Every step back is a setup for a greater comeback!"
        ]
    else:
        encouragements = [
            "Consistency is key! You're steady and strong!",
        ]
    encouragement = random.choice(encouragements)
    return render_template('show.html', current_gpa=current_gpa, predicted_gpa=predicted_gpa, encouragement=encouragement)


if __name__ == '__main__':
    app.run(debug=True)
