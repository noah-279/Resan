import subprocess
import webbrowser
import time

# Step 1: Start Flask backend
subprocess.Popen(["python", "apps1.py"])

# Step 2: Wait for backend to start
time.sleep(3)

# Step 3: Open frontend (served via backend itself, recommended)
webbrowser.open("http://127.0.0.1:5000/")