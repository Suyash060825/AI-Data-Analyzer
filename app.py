"""
AI Data Analyst - Main Application
A Streamlit web app for automated data analysis and AI-powered insights.
"""

import streamlit as st
import pandas as pd
import io

from utils import (
    load_data,
    analyze_data,
    plot_histogram,
    plot_correlation_heatmap,
    plot_boxplot,
    plot_bar_chart,
    generate_ai_insights,
    ask_question,
    generate_report,
)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; color: #e0e0e0; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #2a2f3e;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #1c2132;
        border: 1px solid #2a2f3e;
        border-radius: 10px;
        padding: 12px 16px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #7eb8f7;
        border-left: 4px solid #4a90e2;
        padding-left: 10px;
        margin: 24px 0 12px 0;
    }

    /* Insight card */
    .insight-box {
        background-color: #1a2035;
        border: 1px solid #2e3a55;
        border-radius: 12px;
        padding: 20px 24px;
        line-height: 1.7;
        font-size: 0.95rem;
    }

    /* Chat message */
    .chat-answer {
        background-color: #162030;
        border-left: 3px solid #4a90e2;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin-top: 8px;
        line-height: 1.7;
    }

    /* Subtle divider */
    hr { border-color: #2a2f3e; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 AI Data Analyst")
    st.markdown("Upload a CSV and get instant analysis, charts, and AI insights.")
    st.divider()

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    st.divider()
    st.markdown("**Settings**")
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Required for AI Insights and Q&A. Get yours at console.anthropic.com",
    )

    dark_mode = st.toggle("Dark Mode", value=True)

    st.divider()
    st.markdown(
        "<small>Built as a final-year project · Python + Streamlit</small>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
if uploaded_file is None:
    # Landing state
    st.markdown("# 📊 AI Data Analyst")
    st.markdown("#### Upload a CSV file from the sidebar to get started.")
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🔍 Auto Analysis")
        st.markdown("Summary stats, missing values, correlations — all computed instantly.")
    with col2:
        st.markdown("### 📈 Smart Charts")
        st.markdown("Histograms, heatmaps, boxplots, and bar charts with one click.")
    with col3:
        st.markdown("### 🤖 AI Insights")
        st.markdown("Ask questions about your data in plain English and get precise answers.")
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
df, error = load_data(uploaded_file)

if error:
    st.error(f"❌ Could not load file: {error}")
    st.stop()

if df.empty:
    st.warning("⚠️ The uploaded file is empty.")
    st.stop()

analysis = analyze_data(df)

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 📊 AI Data Analyst")
st.markdown(f"**File:** `{uploaded_file.name}`")
st.divider()

st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{df.shape[0]:,}")
m2.metric("Columns", df.shape[1])
m3.metric("Numeric Cols", len(analysis["numeric_cols"]))
m4.metric("Missing Values", int(analysis["missing"].sum()))

st.markdown("**Preview (first 10 rows)**")
st.dataframe(df.head(10), use_container_width=True)

with st.expander("Column Types"):
    dtype_df = pd.DataFrame({
        "Column": df.dtypes.index,
        "Type": df.dtypes.astype(str).values,
        "Non-Null": df.count().values,
        "Null": df.isnull().sum().values,
    })
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# 2. STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Statistical Analysis</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Summary Stats", "Missing Values", "Top Correlations"])

with tab1:
    if not analysis["stats"].empty:
        st.dataframe(analysis["stats"].round(3), use_container_width=True)
    else:
        st.info("No numeric columns found.")

with tab2:
    missing_df = analysis["missing_df"]
    if missing_df.empty:
        st.success("✅ No missing values found in the dataset.")
    else:
        st.dataframe(missing_df, use_container_width=True, hide_index=True)

with tab3:
    top_corr = analysis.get("top_correlations")
    if top_corr is not None and not top_corr.empty:
        st.dataframe(top_corr, use_container_width=True, hide_index=True)
    else:
        st.info("Need at least 2 numeric columns for correlation.")

# ══════════════════════════════════════════════════════════════════════════════
# 3. VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Visualizations</div>', unsafe_allow_html=True)

viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
    "📊 Histogram", "🔥 Heatmap", "📦 Boxplot", "📋 Bar Chart"
])

with viz_tab1:
    if analysis["numeric_cols"]:
        col = st.selectbox("Select numeric column", analysis["numeric_cols"], key="hist_col")
        fig = plot_histogram(df, col)
        st.pyplot(fig)
    else:
        st.info("No numeric columns available.")

with viz_tab2:
    if len(analysis["numeric_cols"]) >= 2:
        fig = plot_correlation_heatmap(df, analysis["numeric_cols"])
        st.pyplot(fig)
    else:
        st.info("Need at least 2 numeric columns to plot a heatmap.")

with viz_tab3:
    if analysis["numeric_cols"]:
        col = st.selectbox("Select column for boxplot", analysis["numeric_cols"], key="box_col")
        fig = plot_boxplot(df, col)
        st.pyplot(fig)
    else:
        st.info("No numeric columns available.")

with viz_tab4:
    if analysis["categorical_cols"]:
        col = st.selectbox("Select categorical column", analysis["categorical_cols"], key="bar_col")
        top_n = st.slider("Show top N categories", 3, 20, 10)
        fig = plot_bar_chart(df, col, top_n)
        st.pyplot(fig)
    else:
        st.info("No categorical columns found.")

# ══════════════════════════════════════════════════════════════════════════════
# 4. AI INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🤖 AI Insights</div>', unsafe_allow_html=True)

if not api_key:
    st.warning("Enter your Anthropic API key in the sidebar to unlock AI features.")
else:
    if st.button("Generate Insights", type="primary"):
        with st.spinner("Analysing your dataset…"):
            insights, err = generate_ai_insights(df, analysis, api_key)
        if err:
            st.error(f"API error: {err}")
        else:
            st.markdown(
                f'<div class="insight-box">{insights}</div>',
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# 5. ASK A QUESTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">💬 Ask a Question</div>', unsafe_allow_html=True)

if not api_key:
    st.warning("Enter your Anthropic API key in the sidebar to use the Q&A feature.")
else:
    question = st.text_input(
        "Ask anything about the dataset",
        placeholder="e.g. Which features are most correlated with price?",
    )
    if question:
        with st.spinner("Thinking…"):
            answer, err = ask_question(df, analysis, question, api_key)
        if err:
            st.error(f"API error: {err}")
        else:
            st.markdown(
                f'<div class="chat-answer">{answer}</div>',
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# 6. DOWNLOAD REPORT
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="section-header">⬇️ Download Report</div>', unsafe_allow_html=True)

report_md = generate_report(df, analysis, uploaded_file.name)
st.download_button(
    label="Download Report (.md)",
    data=report_md,
    file_name="data_analysis_report.md",
    mime="text/markdown",
)
