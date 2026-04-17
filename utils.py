import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from openai import OpenAI

# =========================
# LOAD DATA (CACHED)
# =========================
def load_data(file):
    try:
        df = pd.read_csv(file)
        return df
    except Exception:
        return None

# =========================
# ANALYSIS
# =========================
def analyze_data(df):
    analysis = {}

    analysis["shape"] = df.shape
    analysis["columns"] = df.columns.tolist()
    analysis["dtypes"] = df.dtypes.astype(str).to_dict()
    analysis["missing"] = df.isnull().sum().to_dict()

    numeric_df = df.select_dtypes(include=np.number)

    if not numeric_df.empty:
        analysis["summary"] = numeric_df.describe().to_dict()
        analysis["correlation"] = numeric_df.corr().to_dict()

    return analysis

# =========================
# TOP CORRELATIONS
# =========================
def get_top_correlations(df):
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] < 2:
        return []

    corr = num_df.corr()
    pairs = []

    for i in corr.columns:
        for j in corr.columns:
            if i != j:
                pairs.append((i, j, corr.loc[i, j]))

    pairs = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
    return pairs[:5]

# =========================
# PLOTS
# =========================
def plot_histogram(df, col):
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f"Distribution of {col}")
    return fig

def plot_box(df, col):
    fig, ax = plt.subplots()
    sns.boxplot(x=df[col], ax=ax)
    ax.set_title(f"Boxplot of {col}")
    return fig

def plot_bar(df, col):
    fig, ax = plt.subplots()
    df[col].value_counts().head(10).plot(kind="bar", ax=ax)
    ax.set_title(f"Top Categories in {col}")
    return fig

def plot_heatmap(df):
    num_df = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig

# =========================
# AI CLIENT
# =========================
def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

# =========================
# AI INSIGHTS
# =========================
def generate_ai_insights(df):
    client = get_client()
    if client is None:
        return "⚠️ Add OPENAI_API_KEY to use AI insights."

    summary = df.describe().to_string()
    cols = ", ".join(df.columns)

    prompt = f"""
You are a data analyst.

Dataset columns: {cols}

Summary:
{summary}

Give:
1. Key insights (specific numbers)
2. Trends
3. Anomalies
4. Suggestions

Be concise and specific.
"""

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content

# =========================
# ASK QUESTIONS
# =========================
def ask_question(df, question):
    client = get_client()
    if client is None:
        return "⚠️ Add API key."

    summary = df.describe().to_string()

    prompt = f"""
You are analyzing a dataset.

Summary:
{summary}

Question: {question}

Answer clearly using dataset context.
"""

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content
