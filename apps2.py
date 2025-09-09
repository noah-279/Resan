from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
import re
import io
import os
from typing import List, Dict, Tuple

# -----------------------
# FLASK APP SETUP
# -----------------------
app = Flask(__name__)
CORS(app)

# -----------------------
# DATABASE SETUP
# -----------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'resumes.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database Model
class ResumeData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200))
    email = db.Column(db.String(200))
    phone = db.Column(db.String(50))
    job_role = db.Column(db.String(100))
    matched_skills = db.Column(db.String(500))
    resume_accuracy = db.Column(db.Integer)

with app.app_context():
    db.create_all()

# -----------------------
# IMPORTS FOR RESUME PARSING
# -----------------------
try:
    from PyPDF2 import PdfReader
except Exception as e:
    raise RuntimeError("PyPDF2 is required. Install with: pip install PyPDF2")

try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None
except Exception:
    spacy = None
    nlp = None

# -----------------------
# CONSTANTS
# -----------------------
ALLOWED_EXTENSIONS = {"pdf", "txt"}

COMMON_SKILLS = [
    "python", "java", "c++", "c", "sql", "nosql", "javascript", "html", "css",
    "react", "angular", "nodejs", "node", "flask", "django", "git", "docker",
    "kubernetes", "machine learning", "deep learning", "nlp", "computer vision",
    "pytorch", "tensorflow", "data analysis", "pandas", "numpy", "matplotlib",
    "scikit-learn", "excel", "tableau", "aws", "azure", "gcp", "linux", "rest api",
    "api", "microservices", "bash", "communication", "leadership", "project management",
    "agile", "scrum"
]

JOB_PROFILES = {
    "Data Scientist": ["python", "machine learning", "pandas", "numpy", "scikit-learn", "sql", "data analysis"],
    "Machine Learning Engineer": ["python", "pytorch", "tensorflow", "deep learning", "machine learning", "docker"],
    "Data Analyst": ["sql", "excel", "tableau", "data analysis", "python"],
    "Backend Developer": ["python", "django", "flask", "sql", "rest api", "docker"],
    "Frontend Developer": ["javascript", "react", "html", "css"],
    "Full Stack Developer": ["javascript", "react", "nodejs", "python", "django", "sql"],
    "DevOps Engineer": ["docker", "kubernetes", "aws", "linux", "git"],
    "Software Engineer": ["java", "c++", "git", "algorithms", "data structures"],
    "Product Manager": ["communication", "leadership", "project management", "agile"]
}

# -----------------------
# HELPER FUNCTIONS
# -----------------------
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf_bytes(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
        return "\n".join(pages)
    except Exception:
        return ""

def extract_text_from_txt_bytes(data: bytes) -> str:
    return data.decode('utf-8', errors='ignore')

EMAIL_RE = re.compile(r"[\w.-]+@[\w.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+?\d{1,3}[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}")

def parse_contact_info(text: str) -> Dict[str, str]:
    email_match = EMAIL_RE.search(text)
    phone_match = PHONE_RE.search(text)
    return {
        "email": email_match.group(0) if email_match else "",
        "phone": phone_match.group(0) if phone_match else "",
    }

def parse_name(text: str) -> str:
    if nlp is not None:
        try:
            doc = nlp(text[:4000])
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    return ent.text
        except Exception:
            pass
    return text.splitlines()[0] if text.splitlines() else ""

def find_skills(text: str, skills_list: List[str]) -> List[str]:
    text_l = text.lower()
    found = []
    for skill in skills_list:
        if re.search(r"\b" + re.escape(skill.lower()) + r"\b", text_l):
            found.append(skill)
    return list(dict.fromkeys(found))  # unique order preserved

def match_target_job(skills_found: List[str], target_job: str) -> Dict:
    if not target_job or target_job not in JOB_PROFILES:
        return {'target_job': target_job or None, 'required_skills': [], 'matched_skills': [], 'missing_skills': [], 'match_score': 0.0}
    required = JOB_PROFILES[target_job]
    matched = [s for s in required if s.lower() in [sf.lower() for sf in skills_found]]
    missing = [s for s in required if s not in matched]
    score = len(matched) / max(1, len(required))
    return {'target_job': target_job, 'required_skills': required, 'matched_skills': matched, 'missing_skills': missing, 'match_score': round(score, 3)}

# -----------------------
# ROUTES
# -----------------------
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/jobs', methods=['GET'])
def list_jobs():
    return jsonify({'jobs': JOB_PROFILES})

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    job_role = request.form.get('job_role', '')

    ext = file.filename.rsplit('.', 1)[1].lower()
    data = file.read()

    if ext == 'pdf':
        text = extract_text_from_pdf_bytes(data)
    else:
        text = extract_text_from_txt_bytes(data)

    name = parse_name(text)
    contact = parse_contact_info(text)
    skills_found = find_skills(text, COMMON_SKILLS)
    job_match = match_target_job(skills_found, job_role)

    resume_accuracy = int(job_match['match_score'] * 100)

    # Save in DB
    resume_entry = ResumeData(
        name=name or "Unknown",
        email=contact.get('email', ''),
        phone=contact.get('phone', ''),
        job_role=job_role,
        matched_skills=", ".join(skills_found),
        resume_accuracy=resume_accuracy
    )
    db.session.add(resume_entry)
    db.session.commit()

    return jsonify({
        "personal_details": {"name": name, "email": contact.get('email'), "phone": contact.get('phone')},
        "skills_found": skills_found,
        "job_match": job_match,
        "resume_accuracy": resume_accuracy,
        "job_role": job_role
    })

@app.route('/all_resumes')
def all_resumes():
    resumes = ResumeData.query.all()
    return jsonify([{
        "id": r.id,
        "name": r.name,
        "email": r.email,
        "phone": r.phone,
        "job_role": r.job_role,
        "skills": r.matched_skills,
        "accuracy": r.resume_accuracy
    } for r in resumes])

# -----------------------
# RUN APP
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)