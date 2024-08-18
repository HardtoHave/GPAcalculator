# GPA Predictor

This project is a web application that predicts a student's GPA based on their academic transcript and other relevant data. The project is written in Python and uses the Flask framework for the web application.

## Modules

The project consists of several modules:

- `app.py`: This is the main module of the Flask application. It handles file uploads, checks for allowed file types, and routes to different pages. It also calculates the current GPA and predicts the next semester's GPA.

- `models/gpa_data_preprocessing.py`: This module loads and preprocesses the student data. It merges several datasets, calculates a GPA-like measure, handles missing values, and encodes categorical variables.

- `models/gpa_predictor.py`: This module predicts the GPA based on the current GPA, courses, and student info features.

- `models/upstage_api.py`: This module uses the OpenAI API to analyze an academic transcript and extract course information.

## Dependencies

The project uses several Python packages, which are listed in the `requirements.txt` file:

- pandas
- sklearn
- joblib
- flask
- werkzeug
- openai
- requests

## Usage

To run the application, use the command `python app.py`. The application will start a local server and you can access the application in your web browser at `http://localhost:5000`.