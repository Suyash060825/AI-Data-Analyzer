# 📊 AI Data Analyst

A Streamlit web application that lets you upload any CSV dataset and instantly get:
- Automated statistical analysis
- Interactive visualisations
- AI-generated insights (powered by Claude)
- Natural language Q&A about your data

---

## 🗂️ Project Structure

```
ai_data_analyst/
├── app.py            # Main Streamlit application
├── utils.py          # Helper functions (analysis, plots, LLM)
├── requirements.txt  # Python dependencies
└── README.md
```

---

## 🚀 Setup & Run

### 1. Clone / download the project

```bash
git clone https://github.com/yourname/ai-data-analyst.git
cd ai_data_analyst
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate.bat       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

---

## 🔑 Anthropic API Key

The AI Insights and Q&A features use the Anthropic Claude API.

1. Get a free API key at [console.anthropic.com](https://console.anthropic.com)
2. Paste the key into the **sidebar input** inside the app
3. No `.env` file needed — the key is entered at runtime

> The key is never stored or logged; it lives only in your browser session.

---

## 📋 Features

| Feature | Description |
|---|---|
| File Upload | CSV upload with preview (first 10 rows) |
| Dataset Info | Shape, column types, null counts |
| Summary Stats | mean, std, min, max, percentiles |
| Missing Values | Count and percentage per column |
| Correlation Matrix | Top correlated feature pairs |
| Histogram | Per-column distribution + KDE |
| Heatmap | Full correlation heatmap |
| Boxplot | Outlier detection per column |
| Bar Chart | Top-N value counts for categorical cols |
| AI Insights | Key findings, trends, anomalies, suggestions |
| Q&A Chat | Ask questions in plain English |
| Download Report | Export analysis as `.md` file |

---

## 🧪 Testing with Sample Data

You can test with any public CSV dataset. Some suggestions:

- [Titanic](https://www.kaggle.com/c/titanic/data)
- [Iris](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv)
- [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** — UI framework
- **Pandas** — data handling
- **Matplotlib + Seaborn** — visualisations
- **Anthropic Claude API** — AI insights
- **Requests** — HTTP calls to LLM

---

## 👤 Author

Built as a final-year Computer Science project.  
Feel free to fork, extend, or submit pull requests.
