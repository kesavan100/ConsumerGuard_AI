# ConsumerGuard AI — Azure App Service Deployment Guide

This guide is written for Azure Student subscriptions.

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Azure Student account | Free $100 credit via [Azure for Students](https://azure.microsoft.com/en-us/free/students/) |
| Azure CLI installed | `winget install Microsoft.AzureCLI` or download from Azure docs |
| Python 3.11 runtime | Selected during App Service creation |
| `chroma_db/` folder | Must be built locally and committed to Git before deploying |

---

## Step 1 — Build the Knowledge Base Locally (REQUIRED FIRST)

```powershell
# From the project root
pip install -r requirements.txt

# Build the ChromaDB from all 9 PDFs
python -m consumerguard.ingest
```

You should see output ending with:
```
✅ Ingestion complete. Total chunks stored: ~300
ChromaDB saved to: ...\chroma_db
```

Verify the `chroma_db/` folder was created and is roughly 5–10 MB.

---

## Step 2 — Copy Your Gemini API Key

You will set this as an Azure App Service environment variable (NOT in code).
Keep your actual key ready.

---

## Step 3 — Create Azure Resources

### Option A: Azure CLI (recommended)

```bash
# Login to Azure
az login

# Create a Resource Group (e.g., in East Asia for low latency with India)
az group create --name ConsumerGuardRG --location eastasia

# Create an App Service Plan (B1 tier for Student subscription)
az appservice plan create \
  --name ConsumerGuardPlan \
  --resource-group ConsumerGuardRG \
  --sku B1 \
  --is-linux

# Create the Web App with Python 3.11
az webapp create \
  --name consumerguard-ai \
  --resource-group ConsumerGuardRG \
  --plan ConsumerGuardPlan \
  --runtime "PYTHON:3.11"
```

> **Note**: Replace `consumerguard-ai` with a globally unique name.
> Your app will be at: `https://consumerguard-ai.azurewebsites.net`

---

## Step 4 — Set the Gemini API Key

```bash
az webapp config appsettings set \
  --name consumerguard-ai \
  --resource-group ConsumerGuardRG \
  --settings GEMINI_API_KEY="your_actual_api_key_here"
```

---

## Step 5 — Configure the Startup Command

```bash
az webapp config set \
  --name consumerguard-ai \
  --resource-group ConsumerGuardRG \
  --startup-file startup.sh
```

---

## Step 6 — Deploy the Application

### Using ZIP deploy (simplest for students)

```powershell
# From the project root — compress the project
Compress-Archive -Path . -DestinationPath consumerguard.zip -Force

# Deploy
az webapp deployment source config-zip \
  --name consumerguard-ai \
  --resource-group ConsumerGuardRG \
  --src consumerguard.zip
```

---

## Step 7 — Verify Deployment

```bash
az webapp log tail \
  --name consumerguard-ai \
  --resource-group ConsumerGuardRG
```

Open your browser: `https://consumerguard-ai.azurewebsites.net`

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError` | Check `requirements.txt` is in the root |
| `Collection not found` | `chroma_db/` missing — re-run ingest, commit, redeploy |
| `GEMINI_API_KEY not set` | Set it via Azure App Settings (Step 4) |
| App returns 503 | Still starting — wait 2 minutes and refresh |

---

## Cost Estimate (Azure Student)

| Resource | Tier | Monthly Cost |
|----------|------|-------------|
| App Service Plan | B1 Linux | ~$13/month |
| Student Credit | $100 free | Covers ~7 months |

---

## Alternative: Deploying via GitHub (Continuous Deployment)

If your code is hosted on GitHub, you can set up continuous deployment so that every time you push to the `main` branch, Azure automatically updates your app.

### Step 1: Push code to GitHub
Make sure your entire project (including the `chroma_db/` folder and `startup.sh`) is pushed to a GitHub repository. 
*(Make sure you do **NOT** commit your `.env` file containing the Gemini API key).*

### Step 2: Connect Azure to GitHub
1. Go to the [Azure Portal](https://portal.azure.com) and navigate to your App Service (`consumerguard-ai`).
2. On the left sidebar, scroll down to **Deployment** and click **Deployment Center**.
3. Under **Source**, select **GitHub**.
4. Authorize Azure to access your GitHub account if prompted.
5. Select your **Organization**, **Repository**, and the **Branch** (usually `main`).
6. Click **Save** at the top.

### Step 3: GitHub Actions will take over
Azure will automatically generate a GitHub Actions workflow file (in your repository under `.github/workflows/`) and trigger a deployment. 
You can track the progress of the build and deployment by going to the **Actions** tab in your GitHub repository!
