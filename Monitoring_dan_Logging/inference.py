import requests
import json
import time
import random
import sys
import logging
from typing import Dict

API_URL = "http://127.0.0.1:8000/predict"

COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
    "amount_bin"
]

logging.basicConfig(
    filename="inference.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

def generate_payload():
    return {
        "inputs": [
            {
                "name": "input-0",
                "shape": [1, 9],
                "datatype": "FP32",
                "data": [[
                    random.randint(1, 1000),
                    random.randint(0, 4),
                    random.uniform(-2, 2),
                    random.uniform(-2, 2),
                    random.uniform(-2, 2),
                    random.uniform(-2, 2),
                    random.uniform(-2, 2),
                    0,
                    random.randint(0, 3)
                ]]
            }
        ]
    }

def send_inference():
    payload = generate_payload()

    try:
        start_time = time.time()
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        latency = time.time() - start_time

        if response.status_code == 200:
            prediction = response.json()["predictions"][0]
            logger.info(
                "Inference success | latency=%.3fs | prediction=%s",
                latency,
                prediction
            )
        else:
            logger.error(
                "Inference failed | status=%s | response=%s",
                response.status_code,
                response.text
            )

    except Exception as e:
        logger.exception("Inference exception: %s", e)


def run_traffic(interval=(0.5, 2.0)):
    logger.info("Traffic generator started")
    try:
        while True:
            send_inference()
            time.sleep(random.uniform(*interval))
    except KeyboardInterrupt:
        logger.info("Traffic generator stopped")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "traffic":
        run_traffic()
    else:
        send_inference()
