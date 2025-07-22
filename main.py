import sys
from flask import Flask
import socket
import logging

# Ensure sys.stdout is line-buffered and UTF-8 encoded
try:
    # Python 3.7+ approach
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")  # type: ignore
    else:
        # Fallback for older Python versions
        sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1, encoding='utf-8', closefd=False)
except (AttributeError, OSError):
    # If reconfigure fails, use the fallback approach
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1, encoding='utf-8', closefd=False)

def find_available_port(start_port: int = 8081, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_attempts}")

from routes import register_routes
from api_routes import register_api_routes

# Configure logging once
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)

register_routes(app)
register_api_routes(app)

if __name__ == "__main__":
    # Find available port starting from 8090
    port = find_available_port(start_port=8090, max_attempts=20)
    logging.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
