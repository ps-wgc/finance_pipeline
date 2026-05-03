"""
server.py - Simple Flask server
Run: python server.py
Then open: http://localhost:8080
"""

from flask import Flask, request, jsonify, send_from_directory
from pipeline import run_pipeline
import os

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('.', 'ui.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    company = data.get('company', '').strip()
    if not company:
        return jsonify({'error': 'Company name is required'}), 400
    try:
        report = run_pipeline(company)
        return jsonify(report)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Open http://localhost:8080 in your browser")
    app.run(port=8080, debug=False)
