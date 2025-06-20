# 🚀 Otto-SR Deployment Guide

This guide will walk you through deploying Otto-SR to GitHub and Heroku.

## Prerequisites

- Git installed on your machine
- GitHub account
- Heroku account (free tier works)
- Heroku CLI installed ([Download here](https://devcenter.heroku.com/articles/heroku-cli))

## 📦 Step 1: Prepare Your Local Repository

```bash
# Navigate to your project
cd /Users/matheusrech/Downloads/FileSortSync-2/FolderFounder

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Otto-SR - AI-powered systematic review screening tool"
```

## 🐙 Step 2: Deploy to GitHub

### 2.1 Create GitHub Repository

1. Go to [GitHub](https://github.com)
2. Click the **"+"** icon → **"New repository"**
3. Name it: `otto-sr` (or your preferred name)
4. Make it **Public** (for free GitHub Actions)
5. **DON'T** initialize with README (we already have one)
6. Click **"Create repository"**

### 2.2 Push to GitHub

```bash
# Add GitHub remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/otto-sr.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### 2.3 Add GitHub Secrets

1. Go to your repository on GitHub
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Add these secrets:
   - `ANTHROPIC_API_KEY`: Your Anthropic API key (for Claude Code Actions)
   - `HEROKU_API_KEY`: Your Heroku API key (see below)
   - `HEROKU_APP_NAME`: Your Heroku app name (you'll create this next)
   - `HEROKU_EMAIL`: Your Heroku account email

## 🚢 Step 3: Deploy to Heroku

### 3.1 Install Heroku CLI and Login

```bash
# Login to Heroku
heroku login

# This will open a browser window for authentication
```

### 3.2 Create Heroku App

```bash
# Create a new Heroku app (choose a unique name)
heroku create otto-sr-app

# Note: If the name is taken, try something like otto-sr-yourname
```

### 3.3 Add PostgreSQL Database

```bash
# Add free PostgreSQL addon
heroku addons:create heroku-postgresql:essential-0

# Wait for database to provision (about 1 minute)
heroku pg:wait
```

### 3.4 Set Environment Variables

```bash
# Set your API keys
heroku config:set OPENAI_API_KEY="your-openai-api-key-here"
heroku config:set ANTHROPIC_API_KEY="your-anthropic-api-key-here"

# Optional: Set other LLM provider keys if you have them
heroku config:set GROQ_API_KEY="your-groq-api-key"
heroku config:set TOGETHER_API_KEY="your-together-api-key"

# Set Python runtime
heroku config:set PYTHON_VERSION=3.11.13

# Set environment
heroku config:set ENVIRONMENT=production
```

### 3.5 Deploy to Heroku

```bash
# Add Heroku remote if not already added
heroku git:remote -a otto-sr-app

# Deploy to Heroku
git push heroku main

# This will take 3-5 minutes as it builds both backend and frontend
```

### 3.6 Scale the Application

```bash
# Ensure at least one web dyno is running
heroku ps:scale web=1
```

### 3.7 Open Your App

```bash
# Open the deployed app in your browser
heroku open
```

## 🔑 Step 4: Get Your Heroku API Key

For GitHub Actions to deploy automatically:

```bash
# Get your Heroku API key
heroku auth:token

# Copy this token and add it as HEROKU_API_KEY in GitHub Secrets
```

## ✅ Step 5: Verify Everything Works

1. **Check GitHub Actions**:
   - Go to your GitHub repo → **Actions** tab
   - You should see the CI/CD pipeline running

2. **Check Heroku Logs**:
   ```bash
   heroku logs --tail
   ```

3. **Test the Application**:
   - Visit your Heroku app URL
   - You should see the beautiful landing page
   - Try uploading a test citation file

## 🔧 Troubleshooting

### If deployment fails:

1. **Check build logs**:
   ```bash
   heroku logs --tail
   ```

2. **Check if database is connected**:
   ```bash
   heroku config | grep DATABASE_URL
   ```

3. **Restart the app**:
   ```bash
   heroku restart
   ```

### Common Issues:

- **"No web processes running"**: Run `heroku ps:scale web=1`
- **Database connection error**: Check DATABASE_URL is set correctly
- **Module not found**: Ensure all dependencies are in requirements.txt
- **Port binding error**: Make sure code uses `$PORT` environment variable

## 🎯 Next Steps

1. **Custom Domain** (Optional):
   ```bash
   heroku domains:add www.yourdomain.com
   ```

2. **Enable Automatic Deploys** (Optional):
   - Go to Heroku Dashboard → Your App → Deploy tab
   - Connect to GitHub
   - Enable automatic deploys from main branch

3. **Monitor Your App**:
   ```bash
   # View metrics
   heroku metrics

   # View logs
   heroku logs --tail
   ```

## 📱 Share Your App

Your app is now live at: `https://otto-sr-app.herokuapp.com`

Share it with the world! 🎉

---

Need help? Check the logs with `heroku logs --tail` or open an issue on GitHub.