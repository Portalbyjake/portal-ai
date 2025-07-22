import os
import signal
import subprocess
import logging
from typing import Optional

from langdetect import detect

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def classify_task(prompt):
    prompt = prompt.lower()
    if "translate" in prompt:
        return "translate"
    elif "summary" in prompt or "summarize" in prompt:
        return "summarize"
    elif any(w in prompt for w in [
        "image", "draw", "create", "picture", "generate", "sketch", "headshot", "photo", "realistic", "see what"
    ]):
        return "image"
    else:
        return "text"

def kill_process_on_port(port: int) -> bool:
    """Kill any process using the specified port."""
    try:
        # Find processes using the port
        result = subprocess.run(
            ['lsof', '-ti', f':{port}'],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    logging.info(f"Killing process {pid} on port {port}")
                    os.kill(int(pid), signal.SIGKILL)
            return True
        return False
    except Exception as e:
        logging.warning(f"Failed to kill process on port {port}: {e}")
        return False

def is_port_in_use(port: int) -> bool:
    """Check if a port is currently in use."""
    try:
        result = subprocess.run(['lsof', '-i', f':{port}'], capture_output=True)
        return result.returncode == 0
    except Exception:
        return False

def get_server_status() -> dict:
    """Get current server status information."""
    status = {
        'port_8081_in_use': is_port_in_use(8081),
        'python_processes': []
    }
    
    try:
        # Get all Python processes
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'python' in line and 'main.py' in line:
                status['python_processes'].append(line.strip())
    except Exception:
        pass
    
    return status
