#!/bin/bash

# Otto-SR Startup Script
echo "🚀 Starting Otto-SR Application..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to kill processes on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

if ! command_exists python3; then
    echo -e "${RED}❌ Python 3 not found! Please install Python 3.11+${NC}"
    exit 1
fi

if ! command_exists npm; then
    echo -e "${RED}❌ npm not found! Please install Node.js${NC}"
    exit 1
fi

if ! command_exists psql; then
    echo -e "${YELLOW}⚠️  PostgreSQL client not found. Database operations may fail.${NC}"
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file from example...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}⚠️  Please update .env with your API keys and database credentials${NC}"
    sleep 3
fi

# Install Python dependencies
echo -e "\n${BLUE}Installing backend dependencies...${NC}"
pip install -q fastapi uvicorn sqlalchemy psycopg2-binary langchain-core langchain-openai aiofiles python-multipart pydantic anthropic python-dotenv

# Install frontend dependencies
echo -e "\n${BLUE}Installing frontend dependencies...${NC}"
cd frontend
npm install --silent
cd ..

# Start PostgreSQL with Docker if available
if command_exists docker; then
    echo -e "\n${BLUE}Starting PostgreSQL with Docker...${NC}"
    docker-compose up -d postgres
    sleep 5
else
    echo -e "${YELLOW}⚠️  Docker not found. Make sure PostgreSQL is running locally.${NC}"
fi

# Start backend
echo -e "\n${GREEN}Starting backend server on http://localhost:8000...${NC}"
python3 main.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start frontend
echo -e "\n${GREEN}Starting frontend on http://localhost:3000...${NC}"
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Display status
echo -e "\n${GREEN}✅ Otto-SR is running!${NC}"
echo -e "${BLUE}Backend:${NC} http://localhost:8000"
echo -e "${BLUE}Frontend:${NC} http://localhost:3000"
echo -e "${BLUE}API Docs:${NC} http://localhost:8000/docs"
echo -e "\n${YELLOW}Press Ctrl+C to stop all services${NC}"

# Keep script running
wait