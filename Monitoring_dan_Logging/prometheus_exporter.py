from flask import Flask, request, jsonify, Response
#import requests
import time
import json
import psutil
import random

from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST
)

app = Flask(__name__)
MLFLOW_MODEL_URL = "http://127.0.0.1:5002/v2/models/fraud-detection-model/infer"

REQUEST_COUNT = Counter(
    "ml_requests_total", "Total inference requests"
)

REQUEST_LATENCY = Histogram(
    "ml_request_latency_seconds", "Total request latency"
)

MODEL_LATENCY = Histogram(
    "ml_model_latency_seconds", "Model inference latency"
)

REQUEST_SIZE = Histogram(
    "ml_request_payload_bytes", "Request payload size"
)

RESPONSE_SIZE = Histogram(
    "ml_response_payload_bytes", "Response payload size"
)

MODEL_SUCCESS = Counter(
    "ml_model_success_total", "Successful predictions"
)

MODEL_ERROR = Counter(
    "ml_model_error_total", "Failed predictions"
)

PREDICTION_CLASS = Counter(
    "ml_prediction_class_total",
    "Prediction per class",
    ["label"]
)

CPU_USAGE = Gauge(
    "system_cpu_usage_percent", "CPU usage percent"
)

MEMORY_USAGE = Gauge(
    "system_memory_usage_percent", "Memory usage percent"
)

UPTIME = Gauge(
    "service_uptime_seconds", "Service uptime"
)

START_TIME = time.time()


@app.route("/metrics")
def metrics():
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().percent)
    UPTIME.set(time.time() - START_TIME)

    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST
    )


@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()

    payload = request.get_json()
    REQUEST_SIZE.observe(len(json.dumps(payload).encode()))

    model_latency = random.uniform(0.05, 0.2)
    time.sleep(model_latency)

    MODEL_LATENCY.observe(model_latency)
    REQUEST_LATENCY.observe(time.time() - start_time)

    prediction = random.choice([0, 1])

    MODEL_SUCCESS.inc()
    PREDICTION_CLASS.labels(str(prediction)).inc()

    result = {"predictions": [prediction]}
    RESPONSE_SIZE.observe(len(json.dumps(result).encode()))

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
