"""
utils.py - Helper functions for AI Data Analyst
Handles data loading, analysis, plotting, and LLM integration.
"""

import io
import json
import textwrap
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import streamlit as st

matplotlib.use("Agg")  # Non-interactive backend for Streamlit

# ── Shared plot style ─────────────────────────────────────────────────────────
DARK_BG = "#0f1117"
PANEL_BG = "#1c2132"
TEXT_COLOR = "#d0d8f0"
ACCENT = "#4a90e2"
GRID_COLOR = "#2a2f3e"


def _apply_dark_style(ax, title="", xlabel="", ylabel=""):
    """Apply consistent dark styling to a Matplotlib axes."""
    ax.set_facecolor(PANEL_BG)
    ax.figure.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.6, alpha=0.7)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> tuple[pd.DataFrame, str | None]:
    """
    Load a CSV from a Streamlit UploadedFile object.
    Returns (dataframe, error_message). error_message is None on success.
    """
    try:
        df = pd.read_csv(uploaded_file)
        # Normalise column names: strip whitespace
        df.columns = df.columns.str.strip()
        return df, None
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), "The file has no data."
    except pd.errors.ParserError as e:
        return pd.DataFrame(), f"Could not parse CSV: {e}"
    except Exception as e:
        return pd.DataFrame(), str(e)


# ══════════════════════════════════════════════════════════════════════════════
# DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def analyze_data(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics, missing values, and correlations.
    Returns a dict used throughout the app.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Summary stats for numeric columns
    stats = df[numeric_cols].describe().T if numeric_cols else pd.DataFrame()

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Column": missing.index,
        "Missing Count": missing.values,
        "Missing %": missing_pct.values,
    })
    missing_df = missing_df[missing_df["Missing Count"] > 0].reset_index(drop=True)

    # Correlation matrix and top pairs
    corr_matrix = None
    top_correlations = None
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        # Extract upper triangle pairs
        pairs = (
            corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            .stack()
            .reset_index()
        )
        pairs.columns = ["Feature A", "Feature B", "Correlation"]
        pairs["Abs Correlation"] = pairs["Correlation"].abs()
        top_correlations = (
            pairs.sort_values("Abs Correlation", ascending=False)
            .drop(columns="Abs Correlation")
            .head(10)
            .reset_index(drop=True)
        )
        top_correlations["Correlation"] = top_correlations["Correlation"].round(4)

    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "stats": stats,
        "missing": missing,
        "missing_df": missing_df,
        "corr_matrix": corr_matrix,
        "top_correlations": top_correlations,
    }


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_histogram(df: pd.DataFrame, column: str):
    """Distribution histogram with KDE overlay for a numeric column."""
    fig, ax = plt.subplots(figsize=(9, 4))
    data = df[column].dropna()

    ax.hist(data, bins=30, color=ACCENT, alpha=0.7, edgecolor=DARK_BG, linewidth=0.5)

    # KDE overlay
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        x = np.linspace(data.min(), data.max(), 300)
        ax2 = ax.twinx()
        ax2.plot(x, kde(x), color="#f5a623", linewidth=2)
        ax2.set_ylabel("Density", color=TEXT_COLOR, fontsize=9)
        ax2.tick_params(colors=TEXT_COLOR)
        ax2.set_facecolor(PANEL_BG)
        for spine in ax2.spines.values():
            spine.set_edgecolor(GRID_COLOR)
    except ImportError:
        pass  # scipy not installed — skip KDE

    _apply_dark_style(ax, title=f"Distribution of {column}", xlabel=column, ylabel="Count")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list):
    """Annotated correlation heatmap for all numeric columns."""
    corr = df[numeric_cols].corr()
    n = len(numeric_cols)
    fig_size = max(7, n * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))

    mask = np.triu(np.ones_like(corr, dtype=bool))  # hide upper triangle
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor=DARK_BG,
        ax=ax,
        annot_kws={"size": 8 if n <= 10 else 6},
        cbar_kws={"shrink": 0.8},
    )

    ax.set_facecolor(PANEL_BG)
    fig.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.set_title("Correlation Matrix", color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    return fig


def plot_boxplot(df: pd.DataFrame, column: str):
    """Boxplot to visualise outliers for a numeric column."""
    fig, ax = plt.subplots(figsize=(9, 4))
    data = df[column].dropna()

    bp = ax.boxplot(
        data,
        vert=False,
        patch_artist=True,
        notch=False,
        widths=0.5,
        boxprops=dict(facecolor=ACCENT, color=ACCENT, alpha=0.6),
        medianprops=dict(color="#f5a623", linewidth=2),
        whiskerprops=dict(color=TEXT_COLOR, linewidth=1.2),
        capprops=dict(color=TEXT_COLOR, linewidth=1.2),
        flierprops=dict(marker="o", color="#e74c3c", alpha=0.5, markersize=4),
    )

    _apply_dark_style(ax, title=f"Boxplot — {column}", xlabel=column)
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def plot_bar_chart(df: pd.DataFrame, column: str, top_n: int = 10):
    """Horizontal bar chart for value counts of a categorical column."""
    counts = df[column].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.4)))

    colors = sns.color_palette("Blues_r", len(counts))
    bars = ax.barh(counts.index.astype(str), counts.values, color=colors, edgecolor=DARK_BG)

    # Add value labels inside bars
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_width() * 0.97, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", ha="right",
            color="white", fontsize=8, fontweight="bold",
        )

    _apply_dark_style(ax, title=f"Top {top_n} — {column}", xlabel="Count", ylabel=column)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# LLM HELPERS  (Anthropic Claude)
# ══════════════════════════════════════════════════════════════════════════════

def _call_claude(prompt: str, api_key: str, max_tokens: int = 800) -> tuple[str, str | None]:
    """
    Call the Anthropic Messages API.
    Returns (response_text, error). error is None on success.
    """
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": "claude-3-5-haiku-20241022",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["content"][0]["text"]
        return text, None
    except requests.exceptions.HTTPError as e:
        try:
            detail = resp.json().get("error", {}).get("message", str(e))
        except Exception:
            detail = str(e)
        return "", detail
    except Exception as e:
        return "", str(e)


def _build_dataset_summary(df: pd.DataFrame, analysis: dict) -> str:
    """Build a concise text summary of the dataset for use in LLM prompts."""
    numeric_cols = analysis["numeric_cols"]
    cat_cols = analysis["categorical_cols"]
    missing_total = int(analysis["missing"].sum())

    lines = [
        f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns.",
        f"Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols) if numeric_cols else 'none'}.",
        f"Categorical columns ({len(cat_cols)}): {', '.join(cat_cols) if cat_cols else 'none'}.",
        f"Total missing values: {missing_total}.",
    ]

    if not analysis["stats"].empty:
        stats = analysis["stats"]
        for col in numeric_cols[:8]:  # cap at 8 to keep prompt short
            row = stats.loc[col]
            lines.append(
                f"  {col}: mean={row['mean']:.3g}, std={row['std']:.3g}, "
                f"min={row['min']:.3g}, max={row['max']:.3g}."
            )

    if analysis["top_correlations"] is not None:
        top = analysis["top_correlations"].head(5)
        for _, r in top.iterrows():
            lines.append(
                f"  Correlation {r['Feature A']} ↔ {r['Feature B']}: {r['Correlation']:.3f}."
            )

    return "\n".join(lines)


def generate_ai_insights(df: pd.DataFrame, analysis: dict, api_key: str) -> tuple[str, str | None]:
    """
    Generate structured AI insights (key findings, trends, anomalies, suggestions).
    """
    summary = _build_dataset_summary(df, analysis)

    prompt = textwrap.dedent(f"""
        You are a senior data analyst. Below is a statistical summary of a dataset.
        Respond ONLY with an HTML-formatted analysis divided into exactly four sections,
        using <b> for bold text. Do not use markdown. Keep each section to 2–4 bullet points.

        Sections to include:
        1. <b>Key Insights</b> — most notable facts about distributions, ranges, or dominant categories.
        2. <b>Trends</b> — patterns, monotonic relationships, or strong correlations between columns.
        3. <b>Anomalies</b> — outliers, unusually high missing rates, suspicious distributions.
        4. <b>Suggestions</b> — concrete next steps (e.g. feature engineering, cleaning, modelling).

        Use actual column names and numeric values from the summary. Be specific, not generic.

        Dataset summary:
        {summary}
    """).strip()

    return _call_claude(prompt, api_key, max_tokens=900)


def ask_question(df: pd.DataFrame, analysis: dict, question: str, api_key: str) -> tuple[str, str | None]:
    """
    Answer a user question about the dataset using the LLM.
    """
    summary = _build_dataset_summary(df, analysis)

    prompt = textwrap.dedent(f"""
        You are a data analyst assistant. A user has asked a question about their dataset.
        Answer concisely (3–6 sentences). Use specific column names, correlations, and
        numeric values from the summary. Do not give generic advice.

        Dataset summary:
        {summary}

        User question: {question}

        Answer:
    """).strip()

    return _call_claude(prompt, api_key, max_tokens=400)


# ══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(df: pd.DataFrame, analysis: dict, filename: str) -> str:
    """Generate a Markdown report summarising the analysis."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Data Analysis Report",
        f"**File:** {filename}  ",
        f"**Generated:** {now}",
        "",
        "---",
        "",
        "## Dataset Overview",
        f"- Rows: {df.shape[0]:,}",
        f"- Columns: {df.shape[1]}",
        f"- Numeric columns: {len(analysis['numeric_cols'])}",
        f"- Categorical columns: {len(analysis['categorical_cols'])}",
        f"- Total missing values: {int(analysis['missing'].sum())}",
        "",
        "## Column Names",
        ", ".join(df.columns.tolist()),
        "",
    ]

    if not analysis["stats"].empty:
        lines += [
            "## Summary Statistics",
            "",
            analysis["stats"].round(3).to_markdown(),
            "",
        ]

    if not analysis["missing_df"].empty:
        lines += [
            "## Missing Values",
            "",
            analysis["missing_df"].to_markdown(index=False),
            "",
        ]

    if analysis["top_correlations"] is not None and not analysis["top_correlations"].empty:
        lines += [
            "## Top Correlations",
            "",
            analysis["top_correlations"].to_markdown(index=False),
            "",
        ]

    lines += [
        "---",
        "*Generated by AI Data Analyst · Built with Python + Streamlit*",
    ]

    return "\n".join(lines)
