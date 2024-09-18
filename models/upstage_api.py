import requests
from openai import OpenAI
from config import *


def get_layout_analysis(api_key, input_filename):
    url = layout_analysis_url
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"document": open(input_filename, "rb")}
    data = {"ocr": True}
    response = requests.post(url, headers=headers, files=files, data=data).json()
    return response


def extract_academic_transcript(input_filename):
    client = OpenAI(
        api_key=API_KEY,
        base_url=chat_base_url,
    )
    layout_response = get_layout_analysis(api_key=API_KEY, input_filename=input_filename)
    output_html = layout_response["html"]
    stream = client.chat.completions.create(
        model="solar-1-mini-chat",
        messages=[
            {
                "role": "system",
                "content": "You are a professional assistant."
            },
            {
                "role": "user",
                "content": "Please analyze the following academic transcript" + output_html +
                           "Extract all course information from that academic transcript. "
                           "Return the data in the following format without any additional text: "
                           "'Subject Number and Name, Mark, Credit Points'. Each course should be on a new line."
                           "For example: 'COMP1511 Programming Fundamentals, 85, 6'."
                           "ensure the credit points are under 20."
                           "if the mark field is empty, fill 0. "
            }
        ],
        stream=True,
    )
    result = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            result += chunk.choices[0].delta.content
    print(result)

    courses = []
    for line in result.splitlines():
        if line.strip():  # Ignore empty lines
            parts = line.split(',')
            if len(parts) == 3:  # Ensure that we have exactly 3 parts
                course = {
                    'subject_name': parts[0].strip(),
                    'mark': parts[1].strip(),
                    'credit_points': parts[2].strip()
                }
                courses.append(course)

    return courses
