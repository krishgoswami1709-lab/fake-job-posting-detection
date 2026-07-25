"""
Flask Web Application for Fake Job Posting Detection System.
Provides REST API endpoints for real-time inference and model evaluation metrics,
serving an interactive web application.
"""

import os
import json
from flask import Flask, request, jsonify, send_from_directory
from predict_system import FakeJobDetector

app = Flask(__name__, static_folder="static", static_url_path="")

# Initialize global detector instance
try:
    detector = FakeJobDetector()
    print("FakeJobDetector initialized successfully!")
except Exception as e:
    print(f"Warning: Detector failed to initialize: {e}")
    detector = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")


@app.route("/")
def serve_index():
    return send_from_directory("static", "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not detector:
        return jsonify({"error": "Model detector not initialized. Ensure models/ directory contains trained artifacts."}), 500
    
    try:
        job_data = request.get_json(force=True)
        if not job_data:
            return jsonify({"error": "No input JSON provided"}), 400
        
        result = detector.predict(job_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    try:
        report_path = os.path.join(ARTIFACTS_DIR, "model_evaluation_report.json")
        eda_path = os.path.join(ARTIFACTS_DIR, "eda_summary.json")

        report_data = {}
        eda_data = {}

        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                report_data = json.load(f)

        if os.path.exists(eda_path):
            with open(eda_path, "r") as f:
                eda_data = json.load(f)

        return jsonify({
            "evaluation_report": report_data,
            "eda_summary": eda_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Fake Job Detector Web App on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
