import json
from datetime import datetime

LOG_FILE = "usage_logs.jsonl"

def log_interaction(data):
    data["timestamp"] = datetime.utcnow().isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")
