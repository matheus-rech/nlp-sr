.PHONY: help build up down restart logs shell clean test migrate

# Default target
help:
	@echo "Otto-SR Docker Commands:"
	@echo "  make build    - Build Docker images"
	@echo "  make up       - Start all services"
	@echo "  make down     - Stop all services"
	@echo "  make restart  - Restart all services"
	@echo "  make logs     - View logs from all services"
	@echo "  make shell    - Open shell in backend container"
	@echo "  make clean    - Remove containers and volumes"
	@echo "  make test     - Run tests"
	@echo "  make migrate  - Run database migrations"

# Build Docker images
build:
	docker-compose build

# Start services
up:
	docker-compose up -d
	@echo "Otto-SR is running!"
	@echo "Frontend: http://localhost:3000"
	@echo "Backend API: http://localhost:8000"
	@echo "API Docs: http://localhost:8000/docs"

# Stop services
down:
	docker-compose down

# Restart services
restart: down up

# View logs
logs:
	docker-compose logs -f

# Open shell in backend container
shell:
	docker-compose exec backend /bin/bash

# Clean up
clean:
	docker-compose down -v
	docker system prune -f

# Run tests
test:
	docker-compose exec backend pytest

# Run database migrations
migrate:
	docker-compose exec backend alembic upgrade head

# Development commands
dev-backend:
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd frontend && npm run dev