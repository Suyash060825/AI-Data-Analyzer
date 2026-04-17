import streamlit as st
import pandas as pd
from utils import *

st.set_page_config(page_title="AI Data Analyst", layout="wide")

st.title("📊 AI Data Analyst")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Upload Data")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# =========================
# MAIN LOGIC
# =========================
if file:
    df = load_data(file)

    if df is None:
        st.error("Invalid file")
        st.stop()

    st.success("File uploaded successfully")

    # =====================
    # DATA PREVIEW
    # =====================
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # =====================
    # ANALYSIS
    # =====================
    analysis = analyze_data(df)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Column Types")
        st.write(analysis["dtypes"])

    with col2:
        st.subheader("Missing Values")
        st.write(analysis["missing"])

    if "summary" in analysis:
        st.subheader("Summary Statistics")
        st.dataframe(pd.DataFrame(analysis["summary"]))

    # =====================
    # CORRELATION
    # =====================
    st.subheader("Top Correlations")
    top_corr = get_top_correlations(df)
    for a, b, r in top_corr:
        st.write(f"{a} ↔ {b} = {round(r,3)}")

    st.pyplot(plot_heatmap(df))

    # =====================
    # VISUALS
    # =====================
    st.subheader("Visualizations")

    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include="object").columns

    if len(num_cols) > 0:
        col = st.selectbox("Select numeric column", num_cols)

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_histogram(df, col))
        with c2:
            st.pyplot(plot_box(df, col))

    if len(cat_cols) > 0:
        col = st.selectbox("Select categorical column", cat_cols)
        st.pyplot(plot_bar(df, col))

    # =====================
    # AI INSIGHTS
    # =====================
    st.subheader("AI Insights")

    if st.button("Generate Insights"):
        insights = generate_ai_insights(df)
        st.write(insights)

    # =====================
    # CHAT
    # =====================
    st.subheader("Ask Questions")

    question = st.text_input("Ask about your data")

    if st.button("Ask"):
        if question:
            answer = ask_question(df, question)
            st.write(answer)

    # =====================
    # DOWNLOAD REPORT
    # =====================
    st.subheader("Download Report")

    report = df.describe().to_csv()
    st.download_button("Download Summary CSV", report, "report.csv")

else:
    st.info("Upload a CSV file to begin.")
