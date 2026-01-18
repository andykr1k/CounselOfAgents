.PHONY: help install install-dev test run shell docker-build docker-run docker-cuda clean

# Default target
help:
	@echo "Agent Orchestration System"
	@echo ""
	@echo "Usage:"
	@echo "  make install        Install dependencies"
	@echo "  make install-dev    Install with dev dependencies"
	@echo "  make test           Run tests"
	@echo "  make run            Run in interactive mode"
	@echo "  make shell          Run interactive shell"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build Docker image (CPU)"
	@echo "  make docker-cuda    Build Docker image (CUDA)"
	@echo "  make docker-run     Run in Docker (CPU)"
	@echo "  make docker-gpu     Run in Docker (CUDA)"
	@echo "  make docker-test    Run tests in Docker"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Clean up cache files"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=counsel --cov-report=term-missing

# Running
run:
	python main.py

shell:
	python main.py -i

# Docker - CPU
docker-build:
	docker build -t counsel-agents:latest -f Dockerfile .

docker-run:
	docker run -it --rm \
		-v $(PWD)/projects:/app/projects \
		-v ~/.cache/huggingface:/root/.cache/huggingface \
		-w /app/projects \
		counsel-agents:latest

# Docker - CUDA
docker-cuda:
	docker build -t counsel-agents:cuda -f Dockerfile.cuda .

docker-gpu:
	docker run -it --rm --gpus all \
		-v $(PWD)/projects:/app/projects \
		-v ~/.cache/huggingface:/root/.cache/huggingface \
		-w /app/projects \
		counsel-agents:cuda

# Docker - Testing
docker-test:
	docker run --rm \
		counsel-agents:latest \
		pytest tests/ -v

docker-test-gpu:
	docker run --rm --gpus all \
		counsel-agents:cuda \
		pytest tests/ -v

# Docker Compose
up:
	docker-compose up -d

up-gpu:
	docker-compose -f docker-compose.cuda.yml up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf build/ dist/ 2>/dev/null || true

# Format code
format:
	black counsel/ tests/ main.py
	ruff check --fix counsel/ tests/ main.py

# Lint
lint:
	ruff check counsel/ tests/ main.py
	black --check counsel/ tests/ main.py
