from flask import Flask, request, render_template_string
from urllib.parse import quote
import requests

app = Flask(__name__)

# HTML form for file upload
UPLOAD_FORM = '''
<!DOCTYPE html>
<html>
<head>
    <title>VirusTotal File Scan</title>
</head>
<body>
    <h1>Scan File with VirusTotal</h1>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file" required>
        <button type="submit">Scan</button>
    </form>
</body>
</html>
'''

VIRUSTOTAL_API_KEY = "b2bdbbf5bd14fc54270ade8d9bda24e145539d10a15e0ca1398551ae2d9d2bb6"

@app.route('/', methods=['GET'])
def home():
    return "Hello World!"


@app.route('/report', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            return 'No file uploaded', 400
            
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400

        # Prepare VirusTotal API request
        headers = {
            "accept": "application/json",
            "x-apikey": VIRUSTOTAL_API_KEY
        }

        # Upload file to VirusTotal
        try:
            response = requests.post(
                "https://www.virustotal.com/api/v3/files",
                files={"file": (file.filename, file.stream, file.content_type)},
                headers=headers
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return f'API Error: {str(e)}', 500

        # Get analysis ID
        analysis_id = response.json().get('data', {}).get('id')
        if not analysis_id:
            return 'Failed to get analysis ID', 500

        # Get analysis report
        try:
            report_url = f"https://www.virustotal.com/api/v3/analyses/{quote(analysis_id)}"
            report_response = requests.get(report_url, headers=headers)
            report_response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return f'Report Error: {str(e)}', 500

        return report_response.json()

    return render_template_string(UPLOAD_FORM)

if __name__ == '__main__':
    app.run(debug=True)