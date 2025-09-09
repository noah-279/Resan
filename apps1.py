



from flask import Flask, request, jsonify, render_template 
from flask_cors import CORS
from werkzeug.utils import secure_filename
import re
import math
import io
from typing import List, Dict, Tuple

#Try to import PyPDF2 and optional spaCy


try:
    from PyPDF2 import PdfReader
except Exception as e:
    raise RuntimeError("PyPDF2 is required. Install with: pip install pyPDF2")

try:
    import spacy
    nlp = None
    try:
        # load small model if available
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        # model not installed; skip but keep spacy import available
        nlp = None
except Exception: 
    spacy = None 
    nlp = None

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {"pdf", "txt"}

#A small but practical skill list and job profiles. Extend as needed.

COMMON_SKILLS = [
     "python", "java", "c++", "c", "sql", "nosql", "javascript", "html", "css", 
     "react", "angular", "nodejs", "node", "flask", "django", "git", "docker", "kubernetes", 
     "machine learning", "deep learning", "nlp", "computer vision", "pytorch", "tensorflow", 
     "data analysis", "pandas", "numpy", "matplotlib", "scikit-learn", "excel", "tableau",
     "aws", "azure", "gcp", "linux", "rest api", "api", "microservices", "bash", 
     "communication", "leadership", "project management", "agile", "scrum" 
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


def allowed_file(filename: str) -> bool: 
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf_bytes(data: bytes) -> str: 
    """Extract text from PDF bytes using PyPDF2.""" 
    try: 
        reader = PdfReader(io.BytesIO(data)) 
        pages = [] 
        for page in reader.pages: 
            try: 
                t = page.extract_text() 
                if t: 
                    pages.append(t) 
            except Exception: 
                    continue 
        return "\n".join(pages) 
    except Exception: 
        # fallback: try decode as text 
        try: 
            return data.decode('utf-8', errors='ignore') 
        except Exception: 
            return ""

def extract_text_from_txt_bytes(data: bytes) -> str:
    try: 
        return data.decode('utf-8', errors='ignore') 
    except Exception: 
        return data.decode('latin-1', errors='ignore')

EMAIL_RE = re.compile(r"[\w.-]+@[\w.-]+.[A-Za-z]{2,}") 
PHONE_RE = re.compile(r"(\+?\d{1,3}[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}")

def parse_contact_info(text: str) -> Dict[str, str]: 
    email_match = EMAIL_RE.search(text) 
    phone_match = PHONE_RE.search(text)

    # location/address naive attempt
    location = ""
    for line in text.splitlines()[:30]:  # check top of resume
        l = line.strip()
        if l.lower().startswith("address") or l.lower().startswith("location"):
            parts = l.split(':', 1)
            if len(parts) > 1:
                location = parts[1].strip()
                break

    return {
        "email": email_match.group(0) if email_match else "",
        "phone": phone_match.group(0) if phone_match else "",
        "location": location
    }

def parse_name(text: str) -> str: 
    """Try spaCy NER first, otherwise use a simple heuristic from the top lines.""" 
    if nlp is not None: 
        try: 
            doc = nlp(text[:4000])  # analyze first chunk 
            for ent in doc.ents: 
                if ent.label_ == "PERSON": 
                    # return the first PERSON entity 
                    return ent.text 
        except Exception: 
            pass

    # Heuristic: take first non-empty line that looks like a name (2-4 words, each capitalized)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    candidates = []
    for line in lines[:10]:
        if len(line.split()) in (2, 3, 4):
           words = line.split()
           cap_count = sum(1 for w in words if w[0].isupper())
           if cap_count >= len(words) - 1:
               # avoid emails/phones
               if EMAIL_RE.search(line) or PHONE_RE.search(line):
                   continue
               candidates.append(line)
    return candidates[0] if candidates else ""

SECTION_HEADERS = [
    'summary', 'objective', 'skills', 'technical skills', 'work experience', 'experience', 
    'projects', 'project', 'education', 'certifications', 'achievements', 'personal details', 'contact' 
]

def extract_sections(text: str) -> Dict[str, str]: 
    """Extract sections by looking for common headings. Returns a dict heading -> content.""" 
    lines = [l.rstrip() for l in text.splitlines()] 
    sections = {} 
    current = None 
    buffer = []

    def save_current():
        if current and buffer:
            sections[current] = '\n'.join(buffer).strip()

    for raw in lines:
        line = raw.strip()
        low = line.lower().strip(':')
        # treat short lines with header words as headings
        if any(low == h or low.startswith(h + ' ') or low.endswith(h) for h in SECTION_HEADERS):
            save_current()
            buffer = []
            current = low
            continue
        # also treat lines that are all-caps and short as headings
        if line.isupper() and len(line.split()) <= 4:
            save_current()
            buffer = []
            current = line.lower()
            continue
        if current:
            buffer.append(raw)
    save_current()
    return sections

def find_skills(text: str, skills_list: List[str]) -> List[str]: 
    text_l = text.lower() 
    found = [] 
    for skill in skills_list: 
        sk = skill.lower() # word-boundary match for single words; substring is fine for multiwords 
        pattern = r"\b" + re.escape(sk) + r"\b" 
        if re.search(pattern, text_l): 
            found.append(skill) 
    # remove duplicates and keep original order 
    unique = [] 
    for s in found: 
        if s not in unique: 
            unique.append(s) 
    return unique

def highlight_skills_in_text(text: str, skills_found: List[str]) -> str: 
    """Return HTML where matched skills are wrapped in <mark> tags. Simple and safe for display in web UI.""" 
    if not skills_found: 
        return text 
    text_out = text 
    # sort by length desc to avoid partial overlaps 
    skills_sorted = sorted(skills_found, key=lambda s: -len(s)) 
    for skill in skills_sorted:
         # case-insensitive replace using regex 
         pattern = re.compile(r'(' + re.escape(skill) + r')', flags=re.IGNORECASE) 
         text_out = pattern.sub(r'<mark>\1</mark>', text_out) 
    return text_out

def match_target_job(skills_found: List[str], target_job: str) -> Dict: 
    job = target_job.strip() 
    if not job or job not in JOB_PROFILES:
        return { 
            'target_job': job or None, 
            'required_skills': [], 
            'matched_skills': [], 
            'missing_skills': [], 
            'match_score': 0.0 
        } 
    required = JOB_PROFILES[job] 
    matched = [s for s in required if any(s.lower() == sf.lower() for sf in skills_found)] 
    missing = [s for s in required if s not in matched] 
    match_score = len(matched) / max(1, len(required)) 
    return { 
        'target_job': job, 
        'required_skills': required, 
        'matched_skills': matched, 
        'missing_skills': missing, 
        'match_score': round(match_score, 3) 
    }

def suggest_alternative_jobs(skills_found: List[str], top_n: int = 3, exclude: str = '') -> List[Dict]: 
    scores = [] 
    for job, reqs in JOB_PROFILES.items(): 
        if job == exclude: 
            continue 
        matched = sum(1 for r in reqs if any(r.lower() == sf.lower() for sf in skills_found)) 
        score = matched / max(1, len(reqs)) 
        scores.append((job, round(score, 3))) 
    scores.sort(key=lambda x: -x[1]) 
    return [{'job': s[0], 'score': s[1]} for s in scores[:top_n]]

def compute_resume_accuracy(parsed_sections: Dict[str, str], skills_found: List[str], target_match_score: float, contact_info: Dict[str, str]) -> Tuple[int, Dict]: 
    """Compute a simple resume accuracy score (0-100) and explanation. 
    Factors: 
      - completeness: how many of the important sections are present
      - skill match for target job (0-1 provided) 
      - contact presence (email/phone)
    We weight them so completeness 50%, skills 40%, contact 10%. 
    """ 
    important = ['skills', 'experience', 'projects', 'education'] 
    present = sum(1 for k in important if any(k in s for s in parsed_sections.keys())) 
    completeness = present / len(important)

    contact_present = 1 if (contact_info.get('email') or contact_info.get('phone')) else 0

    score = 0.5 * completeness + 0.4 * float(target_match_score) + 0.1 * contact_present
    percent = int(round(score * 100))
    explanation = {
        'completeness': round(completeness, 3),
        'target_skill_match': round(float(target_match_score), 3),
        'contact_present': bool(contact_present)
    }
    return percent, explanation



@app.route('/jobs', methods=['GET']) 
def list_jobs(): 
    """Return available job profiles and required skills.""" 
    return jsonify({'jobs': JOB_PROFILES})

@app.route('/analyze', methods=['POST']) 
def analyze(): 
    if 'file' not in request.files: 
        return jsonify({'error': 'No file part in the request. Use form-data with key "file".'}), 400 
    file = request.files['file'] 
    if file.filename == '': 
        return jsonify({'error': 'No file selected.'}), 400 
    if not allowed_file(file.filename): 
        return jsonify({'error': f'Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}'}), 400

    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[1].lower()
    data = file.read()

    if ext == 'pdf':
        text = extract_text_from_pdf_bytes(data)
    else:  # txt
        text = extract_text_from_txt_bytes(data)

    text = text.strip()
    if not text:
        return jsonify({'error': 'Could not extract any text from the file.'}), 400

    # parse personal details
    contact = parse_contact_info(text)
    name = parse_name(text)

    # parse sections
    sections = extract_sections(text)
    skills_found = find_skills(text, COMMON_SKILLS)

    # target job matching
    target_job = request.form.get('target_job', '').strip()
    job_match = match_target_job(skills_found, target_job)

    # alternative job suggestions
    alt_jobs = suggest_alternative_jobs(skills_found, top_n=3, exclude=target_job)

    # highlight skills in resume text (HTML)
    highlighted_html = highlight_skills_in_text(text[:5000], skills_found)  # limit to first 5000 chars

    # compute resume accuracy
    resume_percent, explanation = compute_resume_accuracy(sections, skills_found, job_match.get('match_score', 0.0), contact)

    response = {
        'personal_details': {
            'name': name,
            'email': contact.get('email', ''),
            'phone': contact.get('phone', ''),
            'location': contact.get('location', '')
        },
        'skills': {
            'found': skills_found,
            'count': len(skills_found)
        },
        'sections': sections,
        'experience_preview': sections.get('experience') or sections.get('work experience') or '',
        'projects_preview': sections.get('projects') or sections.get('project') or '',
        'job_match': job_match,
        'alternative_jobs': alt_jobs,
        'highlighted_html_preview': highlighted_html,
        'resume_accuracy_percent': resume_percent,
        'accuracy_explanation': explanation
    }

    return jsonify(response)


@app.route('/')
def home():
    # This will look for "index.html" inside the "templates" folder
    return render_template('index.html')

if __name__ == '__main__': 
    # default to 0.0.0.0 for easy testing with a frontend on another device on your network 
    app.run(host='0.0.0.0', port=5000, debug=True)