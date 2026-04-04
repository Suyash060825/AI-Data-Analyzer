# DataLens AI — Data Analyst App

A fully custom web app (no Streamlit) with a professional dark UI built on FastAPI + vanilla HTML/CSS/JS.

## Project Structure

```
analyst_v2/
├── main.py             # FastAPI backend + all API endpoints
├── templates/
│   └── index.html      # Full single-page frontend (HTML/CSS/JS)
├── requirements.txt
└── README.md
```

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the server
python main.py
# or
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/upload` | Upload CSV, returns full analysis |
| GET | `/api/chart/histogram?col=X` | Histogram + KDE |
| GET | `/api/chart/heatmap` | Correlation heatmap |
| GET | `/api/chart/boxplot?col=X` | Boxplot |
| GET | `/api/chart/barchart?col=X&top_n=10` | Bar chart |
| POST | `/api/insights` | AI insights (requires api_key) |
| POST | `/api/ask` | Q&A (requires api_key + question) |

## Tech Stack

- **FastAPI** — Python backend
- **Pandas + NumPy** — data analysis
- **Matplotlib + Seaborn** — charts rendered as PNG → base64
- **Anthropic Claude API** — AI insights + Q&A
- **Vanilla HTML/CSS/JS** — frontend (no framework, no build step)
