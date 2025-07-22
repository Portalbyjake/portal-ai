# Makefile for Portal AI Project

.PHONY: run test setup clean

run:
	@echo "[INFO] Starting server..."
	source venv/bin/activate && python main.py

test:
	@echo "[INFO] Running all tests..."
	source venv/bin/activate && pytest || echo 'pytest not found, running all test_*.py scripts'
	@for f in test_*.py; do \
		echo "[INFO] Running $$f"; \
		source venv/bin/activate && python $$f; \
	done

setup:
	@echo "[INFO] Setting up environment..."
	python3 -m venv venv
	source venv/bin/activate && pip install --upgrade pip
	source venv/bin/activate && pip install -r requirements.txt

clean:
	@echo "[INFO] Cleaning up..."
	rm -rf __pycache__ */__pycache__ .pytest_cache *.log *.pyc *.pyo *.bak *.tmp *.swp debug*.log server.log 