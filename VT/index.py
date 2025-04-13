from flask import Flask, request, render_template_string
from urllib.parse import quote
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:5173",
    "https://gajshield-flask-host.vercel.app"
]}})
# HTML templates
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

RESULTS_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Scan Results</title>
    <style>
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>Scan Results</h1>
    {% if engine_results %}
        <table>
            <tr>
                <th>Antivirus Engine</th>
                <th>Detection Result</th>
            </tr>
            {% for result in engine_results %}
            <tr>
                <td>{{ result.engine_name }}</td>
                <td>{{ result.result or 'No specific result' }}</td>
            </tr>
            {% endfor %}
        </table>
    {% else %}
        <p>No malicious detections found.</p>
    {% endif %}
</body>
</html>
'''

VIRUSTOTAL_API_KEY = "337446e06e512d913af44f157def454a9c6c76a8792f0c4c05b2283c49c6a8c7"

@app.route('/', methods=['GET'])
def home():
    return "Hello World!"

@app.route('/report', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded', 400
            
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400

        headers = {
            "accept": "application/json",
            "x-apikey": VIRUSTOTAL_API_KEY
        }

        try:
            # Upload file
            response = requests.post(
                "https://www.virustotal.com/api/v3/files",
                files={"file": (file.filename, file.stream, file.content_type)},
                headers=headers
            )
            response.raise_for_status()
            analysis_id = response.json().get('data', {}).get('id')
            if not analysis_id:
                return 'Failed to get analysis ID', 500

            # Get report
            report_url = f"https://www.virustotal.com/api/v3/analyses/{quote(analysis_id)}"
            report_response = requests.get(report_url, headers=headers)
            report_response.raise_for_status()
            report_response = report_response.json()
            # Use the dictionary directly; no .json() needed
            report_data = report_response
            
            # Extract engine results
            results = report_data.get('data', {}).get('attributes', {}).get('results', {})
            engine_results = []
            
            for engine in results.values():
                
                # Check if category is not 'undetected' AND result is not None
                if engine.get('category') != 'undetected' and engine.get('result') is not None:
                    engine_results.append({
                        'engine_name': engine.get('engine_name', 'Unknown'),
                        'result': engine.get('result')
                    })
                
            # Return JSON response
            for result in engine_results:
                print(f"Engine: {result['engine_name']} - result: {result['result']}")

            return {'engine_results': engine_results}, 200, {'Content-Type': 'application/json'}

        except requests.exceptions.RequestException as e:
            return f'Error: {str(e)}', 500

    return render_template_string(UPLOAD_FORM)

if __name__ == '__main__':
    app.run(debug=True)