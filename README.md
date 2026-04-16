# FresherForge MVP

A job matching and resume tailoring tool for freshers using LangChain, LangGraph, and Google Gemini.

## Setup

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file in the project root with:
```
GOOGLE_API_KEY=your_google_gemini_api_key
SERPAPI_KEY=your_serpapi_api_key
```

Get your keys from:
- **Google AI Studio (Gemini)**: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
- **SerpAPI**: [https://serpapi.com](https://serpapi.com)

## Running the Project

### Option 1: CLI Test
Run with resume text input:
```powershell
python main.py
```
This will parse resume text and tailored output to console.

### Option 2: Web UI
Launch Streamlit interface:
```powershell
streamlit run main.py
```
Access the web UI at **http://localhost:8501**

Features:
- Upload resume PDF
- Enter job keywords
- View matching jobs
- Get tailored resume

## Project Structure

- `main.py` - Entry point (CLI & Streamlit UI)
- `graph.py` - LanGraph workflow definition
- `agent_state.py` - State schema for the agent
- `tools.py` - Tool implementations (parse, scrape, match, generate)
- `requirements.txt` - Project dependencies
- `.env` - Environment variables (keep secret)

## Workflow

1. **Parse Resume** - Extract skills from PDF/text
2. **Scrape Jobs** - Search for jobs using SerpAPI
3. **Match Jobs** - Find best job matches using FAISS vector search
4. **Generate Resume** - Tailor resume to matched jobs using Gemini 2.0 Flash
