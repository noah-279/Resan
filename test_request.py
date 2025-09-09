import requests
import os

# Ask user for resume file path
resume_path = input("Enter the full path of your resume (e.g. C:\\Users\\anush\\Desktop\\resume.pdf): ").strip()

# Check if file exists
if not os.path.exists(resume_path):
    print("❌ Error: File not found at", resume_path)
    exit()

# Ask user for target job
target_job = input("Enter the target job role (e.g. Data Scientist): ").strip()

url = "http://127.0.0.1:5000/analyze"

# Open file and send request
with open(resume_path, "rb") as f:
    files = {"file": f}
    data = {"target_job": target_job}
    response = requests.post(url, files=files, data=data)

# Print result
if response.status_code == 200:
    print("\n✅ Resume Analysis Result:")
    print(response.json())
else:
    print("\n❌ Error:", response.status_code, response.text)