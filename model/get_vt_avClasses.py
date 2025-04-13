import hashlib
import json
import requests
import time

# VirusTotal API key
VT_API_KEY = "d324b5badbe3f3bcaac15e0d320f3fc16c631cdfc39442071d90d72df2838d71"

def compute_hashes(file_path):
    """Compute MD5, SHA-1, and SHA-256 hashes of a file."""
    md5_hash = hashlib.md5()
    sha1_hash = hashlib.sha1()
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
            sha1_hash.update(byte_block)
            sha256_hash.update(byte_block)
    
    return {
        "md5": md5_hash.hexdigest(),
        "sha1": sha1_hash.hexdigest(),
        "sha256": sha256_hash.hexdigest()
    }

def get_vt_report(file_hash):
    """Query VirusTotal for a file report using its hash."""
    url = f"https://www.virustotal.com/api/v3/files/{file_hash}"
    headers = {"x-apikey": VT_API_KEY}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()  # Report exists
    elif response.status_code == 404:
        return None  # File not found in VirusTotal
    else:
        raise Exception(f"Error querying VirusTotal: {response.status_code}, {response.text}")

def upload_to_vt(file_path):
    """Upload a file to VirusTotal for analysis."""
    url = "https://www.virustotal.com/api/v3/files"
    headers = {"x-apikey": VT_API_KEY}
    with open(file_path, "rb") as file:
        files = {"file": file}
        response = requests.post(url, headers=headers, files=files)
    
    if response.status_code == 200:
        return response.json()  # Analysis ID returned
    else:
        raise Exception(f"Error uploading to VirusTotal: {response.status_code}, {response.text}")

def get_analysis_results(analysis_id):
    """Poll VirusTotal for analysis results using the analysis ID."""
    url = f"https://www.virustotal.com/api/v3/analyses/{analysis_id}"
    headers = {"x-apikey": VT_API_KEY}
    
    while True:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            result = response.json()
            status = result["data"]["attributes"]["status"]
            if status == "completed":
                return result["data"]["attributes"]["results"]  # Detection results
            elif status == "queued" or status == "in-progress":
                print("Analysis still in progress. Waiting...")
                time.sleep(10)  # Wait before polling again
            else:
                raise Exception(f"Unexpected status: {status}")
        else:
            raise Exception(f"Error retrieving analysis results: {response.status_code}, {response.text}")

def process_uploaded_file(file_path):
    """
    Process an uploaded file and generate LDJSON output for AVClass.
    Each line in the output file contains a single JSON object with 'sha256' and 'av_labels'.
    """
    # Step 1: Compute hashes
    hashes = compute_hashes(file_path)
    print(f"Computed hashes: {hashes}")
    
    # Step 2: Check if the file has already been analyzed
    vt_report = get_vt_report(hashes["sha256"])
    if vt_report:
        print("Report found in VirusTotal.")
        vt_results = vt_report["data"]["attributes"]["last_analysis_results"]
    else:
        print("No report found. Uploading file to VirusTotal...")
        # Step 3: Upload the file to VirusTotal
        upload_response = upload_to_vt(file_path)
        analysis_id = upload_response["data"]["id"]
        
        # Step 4: Poll for analysis results
        vt_results = get_analysis_results(analysis_id)
    
    # Step 5: Format AV labels
    av_labels = []
    for av_name, result in vt_results.items():
        if result["category"] == "malicious" or result["category"] == "suspicious":
            av_labels.append([av_name, result["result"]])
    
    # Step 6: Construct the final JSON object
    output_json = {
        "md5": hashes["md5"],
        "sha1": hashes["sha1"],
        "sha256": hashes["sha256"],
        "av_labels": av_labels
    }
    
    # Save the output JSON to a file in LDJSON format
    with open("output_ldjson.json", "w") as f:
        json.dump(output_json, f)
        f.write("\n")  # Add a newline to ensure LDJSON format
    
    print("Output JSON saved to 'output_ldjson.json'.")
    return output_json

# Example usage
if __name__ == "__main__":
    output = process_uploaded_file("JaffaCakes118_b23c693ab0321b30bf3272efb39ef280.exe")
    print("Generated JSON:", json.dumps(output, indent=4))