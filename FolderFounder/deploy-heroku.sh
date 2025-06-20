#!/bin/bash

# Heroku Deployment Script for Otto-SR
echo "🚀 Otto-SR Heroku Deployment Script"
echo "===================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if Heroku CLI is installed
if ! command -v heroku &> /dev/null; then
    echo -e "${RED}❌ Heroku CLI not found!${NC}"
    echo "Please install it from: https://devcenter.heroku.com/articles/heroku-cli"
    exit 1
fi

# Login to Heroku
echo -e "\n${BLUE}Logging into Heroku...${NC}"
heroku login

# Get app name from user
echo -e "\n${YELLOW}Enter your Heroku app name (e.g., otto-sr-yourname):${NC}"
read -r APP_NAME

# Create Heroku app
echo -e "\n${BLUE}Creating Heroku app: $APP_NAME${NC}"
heroku create "$APP_NAME" || {
    echo -e "${YELLOW}App might already exist, continuing...${NC}"
}

# Add PostgreSQL
echo -e "\n${BLUE}Adding PostgreSQL database...${NC}"
heroku addons:create heroku-postgresql:essential-0 --app "$APP_NAME"

# Wait for database
echo -e "\n${BLUE}Waiting for database to provision...${NC}"
heroku pg:wait --app "$APP_NAME"

# Get API keys from user
echo -e "\n${YELLOW}Enter your OpenAI API key:${NC}"
read -r OPENAI_KEY

echo -e "\n${YELLOW}Enter your Anthropic API key:${NC}"
read -r ANTHROPIC_KEY

# Set environment variables
echo -e "\n${BLUE}Setting environment variables...${NC}"
heroku config:set \
    OPENAI_API_KEY="$OPENAI_KEY" \
    ANTHROPIC_API_KEY="$ANTHROPIC_KEY" \
    ENVIRONMENT=production \
    PYTHON_VERSION=3.11.13 \
    --app "$APP_NAME"

# Add git remote
echo -e "\n${BLUE}Adding Heroku git remote...${NC}"
heroku git:remote -a "$APP_NAME"

# Deploy
echo -e "\n${BLUE}Deploying to Heroku (this may take 3-5 minutes)...${NC}"
git push heroku main

# Scale dynos
echo -e "\n${BLUE}Scaling web dyno...${NC}"
heroku ps:scale web=1 --app "$APP_NAME"

# Get app info
echo -e "\n${GREEN}✅ Deployment complete!${NC}"
echo -e "\n${BLUE}Your app is available at:${NC}"
heroku info --app "$APP_NAME" | grep "Web URL"

echo -e "\n${BLUE}To view logs:${NC} heroku logs --tail --app $APP_NAME"
echo -e "${BLUE}To open app:${NC} heroku open --app $APP_NAME"

# Open the app
echo -e "\n${YELLOW}Would you like to open the app now? (y/n)${NC}"
read -r OPEN_APP
if [[ $OPEN_APP == "y" ]]; then
    heroku open --app "$APP_NAME"
fi