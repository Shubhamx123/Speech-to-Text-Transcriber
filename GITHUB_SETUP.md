# GitHub Setup Guide

Follow these steps to push your Speech-to-Text Transcriber to GitHub and deploy it on Streamlit Cloud.

## Step 1: Create a GitHub Repository

1. Go to [GitHub](https://github.com) and sign in to your account
2. Click on the "+" icon in the top right corner and select "New repository"
3. Name your repository (e.g., "Speech-to-Text-Transcriber")
4. Add a description (optional)
5. Choose "Public" visibility (required for Streamlit Cloud free tier)
6. Click "Create repository"

## Step 2: Initialize Git and Push Your Code

Run these commands in your project directory:

```bash
# Initialize Git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit"

# Add GitHub repository as remote
git remote add origin https://github.com/YourUsername/Speech-to-Text-Transcriber.git

# Push to GitHub
git push -u origin main  # or 'master' depending on your default branch name
```

*Note: Replace 'YourUsername' with your actual GitHub username*

## Step 3: Deploy to Streamlit Cloud

1. Go to [Streamlit Cloud](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Choose your repository from the list
4. Configure as follows:
   - Main file path: `main.py`
   - Branch: `main` (or whichever branch you pushed to)
5. Click "Deploy"

Your app will be deployed at a URL like:
`https://username-speech-to-text-transcriber.streamlit.app`

## Step 4: Making Changes After Deployment

Any time you want to update your app:

1. Make changes to your code locally
2. Commit changes:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```
3. Push to GitHub:
   ```bash
   git push
   ```

Streamlit Cloud will automatically detect the changes and rebuild your app. 