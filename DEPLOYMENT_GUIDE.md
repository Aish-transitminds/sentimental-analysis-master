# How to Deploy Sentiment Analysis AI

Since this application uses a powerful AI model (`transformers`), it cannot be hosted on simple static sites (like GitHub Pages) or size-limited serverless functions (like Vercel standard tier).

## Recommended Option: Render.com (Free Tier Available)

### Step 1: Get your code on GitHub
1.  Go to [GitHub.com](https://github.com) and create a new repository named `sentiment-analysis`.
2.  Open the repository page and click **"uploading an existing file"**.
3.  Drag and drop all the files from this folder (`c:\Users\Asus\Documents\Sentiment-Analysis-Using-Flask-master`) into the GitHub upload area.
4.  Commit the changes.

### Step 2: Deploy on Render
1.  Go to [dashboard.render.com](https://dashboard.render.com/) and Sign Up/Login.
2.  Click **"New +"** -> **"Web Service"**.
3.  Select **"Build and deploy from a Git repository"**.
4.  Connect your GitHub account and select your `sentiment-analysis` repository.
5.  **Configure the details**:
    *   **Name**: `sentiment-ai` (or unique name)
    *   **Region**: Any (e.g., Singapore, Frankfurt)
    *   **Branch**: `main` (or master)
    *   **Runtime**: `Python 3`
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `gunicorn app:app`
    *   **Instance Type**: `Free`
6.  Click **"Create Web Service"**.

### Step 3: Wait & Test
*   Render will start building your app. It will install dependencies (torch, transformers, etc.).
*   **Note**: The build might take 5-10 minutes because AI libraries are large.
*   Once it says "Live", click the URL provided by Render.
*   **First Run**: The first time you analyze text, it will download the model. This might take 30-60 seconds. *Be patient on the first try!*

## Troubleshooting
*   **"Memory Error"**: If the free tier runs out of RAM, try a lighter model in `views.py` (e.g., `distilbert-base-uncased-finetuned-sst-2-english`).
*   **"Timeout"**: The free tier puts apps to sleep. The first load after a long time will be slow.
