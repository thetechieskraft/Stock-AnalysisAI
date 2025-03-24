# Stock-Analysis

A **Multi-Agent Stock Analysis App** using **AutoGen 0.4** and **Azure AI Agent Service**.

## Overview

This project leverages **AutoGen**, a framework for building AI agents and applications, along with **Azure AI Agent Service** to perform stock analysis.

🔗 **AutoGen Documentation:** [AutoGen Framework](https://microsoft.github.io/autogen/dev/index.html)

---

## Step-by-Step Configuration

### 1. Set Up Azure AI Hub
1. **Create** an **Azure AI Hub** using the **Azure AI Foundry** service in Azure Cloud.
2. **Launch** Azure AI Foundry.
3. **Create a new project** inside Azure AI Foundry.
4. An **Azure OpenAI Service endpoint** will be created.
5. **Save the Project Connection String** for later use.

### 2. Model Deployment in Azure AI Foundry
6. Navigate to **Models & Endpoints** or **Playgrounds** (recommended).
7. In **Playgrounds**, select **Assistant Playground** (only supported models will be visible).
8. Choose **"gpt-4o"** and select the appropriate model under the **Deployment Details** section.

🔗 **Model Documentation:** [Azure OpenAI Models](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models?tabs=global-standard%2Cstandard-chat-completions)

### 3. Enable Bing Search for Grounding
9. **Create** a **Bing Search Resource** and set up Grounding.
10. **Connect** Bing Search (`https://api.bing.microsoft.com/`) with **AI Hub**.
11. In **Azure AI Management Center**, navigate to **New Resources**.
12. Under **Other Options**, select:
    - **API Key** from Bing Search.
    - **Bing API Endpoint** from [Microsoft Documentation](https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/).

### 4. Configure Environment Variables
13. Set the required **environment variables** for Azure AI Hub and Bing Search.
14. Ensure the **model version** is set to `"2025-01-01-preview"` or you can copy it target URI of model deployment.

---

## Setting Up a Virtual Environment (Recommended)
Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # For Linux/macOS
venv\Scripts\activate      # For Windows

## Install dependencies:

pip install -r requirements.txt

## If installation fails, upgrade pip:

python -m pip install --upgrade pip

## Run the application:

python app.py


Deactivating Virtual Environment

deactivate

Alternative Installation (From PyPi - Recommended)

pip install -U autogenstudio
