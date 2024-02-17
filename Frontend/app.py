import json
from flask import Flask, render_template, request, redirect, url_for
import os
import threading

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

processing_status_ = {'status': 'Waiting for processing...', 'result': None}

def create_folders():
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['OUTPUT_FOLDER']):
        os.makedirs(app.config['OUTPUT_FOLDER'])

create_folders()

def process_file(file_path):
    global processing_status_
    command = f'python predictWithOCR.py model=best.pt source={file_path}'
    result = os.system(command)
    
    with open('Predicts.txt', 'r') as f:
        op = f.readline()
    
        output_json_filepath = os.path.join(app.config['OUTPUT_FOLDER'], 'output.json')

        json_data = json.dumps({"number": op})
        with open(output_json_filepath, 'w') as json_file:
            json_file.write(json_data)

        processing_status_['status'] = 'Processing complete'
        processing_status_['result'] = op
    processing_status()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    process_thread = threading.Thread(target=process_file, args=(file_path,), daemon=True)
    process_thread.start()

    return redirect(url_for('processing_status'))

@app.route('/processing-status')
def processing_status():
    global processing_status_
    with app.app_context():
        return render_template('processing_status.html', status=processing_status_['status'], result=processing_status_['result'])

if __name__ == '__main__':
    app.run(debug=True)
