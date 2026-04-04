"""
AI Data Analyst — FastAPI Backend
Serves the single-page app and all API endpoints.
No external API keys required — insights and Q&A are fully rule-based.
"""

import io
import base64
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats

matplotlib.use("Agg")

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="AI Data Analyst")

# In-memory store keyed by session (single user MVP — extend with Redis for multi-user)
_store: dict = {}

# ── Plot theme ────────────────────────────────────────────────────────────────
BG       = "#0d1117"
SURFACE  = "#161b27"
BORDER   = "#21283b"
TEXT     = "#cdd6f4"
MUTED    = "#6c7a9e"
ACCENT   = "#5b9cf6"
ACCENT2  = "#f5a742"
DANGER   = "#f28b82"

def _fig_style():
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    SURFACE,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   MUTED,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "text.color":        TEXT,
        "grid.color":        BORDER,
        "grid.linewidth":    0.6,
        "axes.grid":         True,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":       "DejaVu Sans",
        "font.size":         10,
    })

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        df.columns = df.columns.str.strip()
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    if df.empty:
        raise HTTPException(400, "File is empty.")

    numeric_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols      = df.select_dtypes(include=["object", "category"]).columns.tolist()
    missing       = df.isnull().sum()
    missing_pct   = (missing / len(df) * 100).round(2)

    # Stats
    stats = {}
    if numeric_cols:
        desc = df[numeric_cols].describe().T
        for col in numeric_cols:
            r = desc.loc[col]
            stats[col] = {
                "mean": round(float(r["mean"]), 4),
                "std":  round(float(r["std"]),  4),
                "min":  round(float(r["min"]),  4),
                "max":  round(float(r["max"]),  4),
                "p25":  round(float(r["25%"]),  4),
                "p50":  round(float(r["50%"]),  4),
                "p75":  round(float(r["75%"]),  4),
            }

    # Missing
    missing_data = [
        {"col": c, "count": int(missing[c]), "pct": float(missing_pct[c])}
        for c in df.columns if missing[c] > 0
    ]

    # Correlations
    top_corr = []
    corr_cols = numeric_cols
    if len(corr_cols) >= 2:
        corr = df[corr_cols].corr()
        pairs = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack().reset_index()
        )
        pairs.columns = ["a", "b", "r"]
        pairs["abs"] = pairs["r"].abs()
        for _, row in pairs.sort_values("abs", ascending=False).head(8).iterrows():
            top_corr.append({"a": row["a"], "b": row["b"], "r": round(float(row["r"]), 4)})

    # Preview
    preview_cols = df.columns.tolist()
    preview_rows = df.head(8).fillna("").values.tolist()

    payload = {
        "filename":     file.filename,
        "rows":         df.shape[0],
        "cols":         df.shape[1],
        "numeric_cols": numeric_cols,
        "cat_cols":     cat_cols,
        "missing_total":int(missing.sum()),
        "stats":        stats,
        "missing_data": missing_data,
        "top_corr":     top_corr,
        "preview_cols": preview_cols,
        "preview_rows": preview_rows,
    }

    # Store df for chart/AI endpoints
    _store["df"]      = df
    _store["payload"] = payload

    return JSONResponse(payload)


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/chart/histogram")
async def histogram(col: str):
    df = _get_df()
    data = df[col].dropna()
    _fig_style()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.hist(data, bins=32, color=ACCENT, alpha=0.85, edgecolor=BG, linewidth=0.4)
    ax.set_title(f"Distribution — {col}", color=TEXT, fontsize=11, pad=10)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        x = np.linspace(data.min(), data.max(), 300)
        ax2 = ax.twinx()
        ax2.plot(x, kde(x), color=ACCENT2, linewidth=2)
        ax2.set_ylabel("Density", color=MUTED)
        ax2.tick_params(colors=MUTED)
        ax2.set_facecolor(SURFACE)
        for s in ax2.spines.values():
            s.set_edgecolor(BORDER)
    except ImportError:
        pass
    fig.tight_layout()
    return JSONResponse({"img": _fig_to_b64(fig)})


@app.get("/api/chart/heatmap")
async def heatmap():
    df = _get_df()
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(cols) < 2:
        raise HTTPException(400, "Need ≥2 numeric columns.")
    corr = df[cols].corr()
    n = len(cols)
    _fig_style()
    fig, ax = plt.subplots(figsize=(max(6, n * 0.7), max(5, n * 0.6)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 15, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, fmt=".2f",
                linewidths=0.4, linecolor=BG, ax=ax,
                annot_kws={"size": 8 if n <= 10 else 6},
                cbar_kws={"shrink": 0.75})
    ax.set_title("Correlation Heatmap", color=TEXT, fontsize=11, pad=10)
    plt.xticks(rotation=40, ha="right")
    fig.tight_layout()
    return JSONResponse({"img": _fig_to_b64(fig)})


@app.get("/api/chart/boxplot")
async def boxplot(col: str):
    df = _get_df()
    data = df[col].dropna()
    _fig_style()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.5,
                    boxprops=dict(facecolor=ACCENT, color=ACCENT, alpha=0.55),
                    medianprops=dict(color=ACCENT2, linewidth=2.2),
                    whiskerprops=dict(color=TEXT, linewidth=1.1),
                    capprops=dict(color=TEXT, linewidth=1.1),
                    flierprops=dict(marker="o", color=DANGER, alpha=0.55, markersize=4))
    ax.set_title(f"Boxplot — {col}", color=TEXT, fontsize=11, pad=10)
    ax.set_xlabel(col)
    ax.set_yticks([])
    fig.tight_layout()
    return JSONResponse({"img": _fig_to_b64(fig)})


@app.get("/api/chart/barchart")
async def barchart(col: str, top_n: int = 10):
    df = _get_df()
    counts = df[col].value_counts().head(top_n)
    _fig_style()
    fig, ax = plt.subplots(figsize=(9, max(3.5, top_n * 0.38)))
    colors = [ACCENT] + [MUTED] * (len(counts) - 1)
    bars = ax.barh(counts.index.astype(str), counts.values, color=colors,
                   edgecolor=BG, linewidth=0.3)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() * 0.98, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", ha="right",
                color="white", fontsize=8, fontweight="bold")
    ax.set_title(f"Top {top_n} — {col}", color=TEXT, fontsize=11, pad=10)
    ax.set_xlabel("Count")
    ax.invert_yaxis()
    fig.tight_layout()
    return JSONResponse({"img": _fig_to_b64(fig)})


# ══════════════════════════════════════════════════════════════════════════════
# RULE-BASED INSIGHTS ENGINE  (no API key required)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/insights")
async def insights():
    df = _get_df()
    payload = _store.get("payload", {})
    result = _generate_insights(df, payload)
    return JSONResponse(result)


@app.post("/api/ask")
async def ask(question: str = Form(...)):
    df = _get_df()
    payload = _store.get("payload", {})
    answer = _answer_question(df, payload, question.lower())
    return JSONResponse({"answer": answer})


def _generate_insights(df: pd.DataFrame, payload: dict) -> dict:
    """
    Fully rule-based insight engine.
    Analyses the dataframe and returns specific, data-driven findings.
    """
    numeric_cols = payload.get("numeric_cols", [])
    cat_cols     = payload.get("cat_cols", [])
    stats        = payload.get("stats", {})
    top_corr     = payload.get("top_corr", [])
    missing_data = payload.get("missing_data", [])

    insights     = []
    trends       = []
    anomalies    = []
    suggestions  = []

    # ── KEY INSIGHTS ──────────────────────────────────────────────────────────
    insights.append(
        f"Dataset has {df.shape[0]:,} rows and {df.shape[1]} columns "
        f"({len(numeric_cols)} numeric, {len(cat_cols)} categorical)."
    )

    # Highest-range numeric column
    if stats:
        ranges = {c: stats[c]["max"] - stats[c]["min"] for c in numeric_cols if c in stats}
        if ranges:
            widest = max(ranges, key=ranges.get)
            insights.append(
                f"'{widest}' has the widest value range: "
                f"{stats[widest]['min']:,} – {stats[widest]['max']:,} "
                f"(mean {stats[widest]['mean']:,})."
            )

    # Most dominant category
    for col in cat_cols[:3]:
        top_val = df[col].value_counts().idxmax()
        top_pct = round(df[col].value_counts(normalize=True).iloc[0] * 100, 1)
        if top_pct > 30:
            insights.append(
                f"'{col}' is dominated by '{top_val}' which appears in {top_pct}% of rows."
            )
            break

    # High-std column (most spread)
    if stats and len(numeric_cols) >= 2:
        cv = {c: stats[c]["std"] / stats[c]["mean"] if stats[c]["mean"] != 0 else 0
              for c in numeric_cols}
        most_varied = max(cv, key=cv.get)
        insights.append(
            f"'{most_varied}' shows the highest relative variability "
            f"(std={stats[most_varied]['std']:,}, mean={stats[most_varied]['mean']:,})."
        )

    # ── TRENDS ────────────────────────────────────────────────────────────────
    if top_corr:
        strongest = top_corr[0]
        direction = "positive" if strongest["r"] > 0 else "negative"
        strength  = "strong" if abs(strongest["r"]) > 0.7 else "moderate"
        trends.append(
            f"Strongest {strength} {direction} correlation: '{strongest['a']}' ↔ "
            f"'{strongest['b']}' (r = {strongest['r']})."
        )

    # Skewness check
    for col in numeric_cols[:4]:
        col_data = df[col].dropna()
        skew = round(float(col_data.skew()), 2)
        if abs(skew) > 1:
            direction = "right (positively)" if skew > 0 else "left (negatively)"
            trends.append(
                f"'{col}' is heavily skewed {direction} (skewness = {skew}), "
                "suggesting outliers or non-normal distribution."
            )
            break

    # Monotonic check using Spearman across top corr pair
    if top_corr and len(top_corr) >= 2:
        c2 = top_corr[1]
        try:
            rho, pval = scipy_stats.spearmanr(
                df[c2["a"]].dropna(), df[c2["b"]].dropna()
            )
            if pval < 0.05:
                trends.append(
                    f"'{c2['a']}' and '{c2['b']}' show a statistically significant "
                    f"monotonic trend (Spearman ρ = {round(rho,3)}, p < 0.05)."
                )
        except Exception:
            pass

    if not trends:
        trends.append("No strong linear trends detected. Consider exploring non-linear relationships.")

    # ── ANOMALIES ─────────────────────────────────────────────────────────────
    # Outlier detection using IQR
    outlier_cols = []
    for col in numeric_cols:
        col_data = df[col].dropna()
        q1, q3   = col_data.quantile(0.25), col_data.quantile(0.75)
        iqr      = q3 - q1
        outliers = ((col_data < q1 - 1.5 * iqr) | (col_data > q3 + 1.5 * iqr)).sum()
        if outliers > 0:
            outlier_cols.append((col, int(outliers)))

    if outlier_cols:
        worst = sorted(outlier_cols, key=lambda x: x[1], reverse=True)[0]
        anomalies.append(
            f"'{worst[0]}' contains {worst[1]} outliers (beyond 1.5×IQR). "
            "Review these rows before modelling."
        )
        if len(outlier_cols) > 1:
            others = ", ".join(f"'{c}' ({n})" for c, n in outlier_cols[1:3])
            anomalies.append(f"Additional columns with outliers: {others}.")

    # Missing values
    if missing_data:
        worst_miss = max(missing_data, key=lambda x: x["pct"])
        anomalies.append(
            f"'{worst_miss['col']}' has the highest missing rate: "
            f"{worst_miss['count']} values ({worst_miss['pct']}% of rows)."
        )
    else:
        anomalies.append("No missing values detected — dataset is complete.")

    # Constant or near-constant columns
    for col in numeric_cols:
        if df[col].nunique() <= 2:
            anomalies.append(
                f"'{col}' has only {df[col].nunique()} unique value(s) — "
                "may not be useful as a feature."
            )
            break

    # ── SUGGESTIONS ───────────────────────────────────────────────────────────
    if missing_data:
        high_miss = [m for m in missing_data if m["pct"] > 30]
        low_miss  = [m for m in missing_data if m["pct"] <= 30]
        if high_miss:
            cols_str = ", ".join(f"'{m['col']}'" for m in high_miss[:3])
            suggestions.append(
                f"Consider dropping {cols_str} — missing rate exceeds 30%."
            )
        if low_miss:
            cols_str = ", ".join(f"'{m['col']}'" for m in low_miss[:3])
            suggestions.append(
                f"Impute missing values in {cols_str} using median (numeric) or mode (categorical)."
            )

    if outlier_cols:
        suggestions.append(
            "Apply log transformation or winsorization to columns with heavy outliers "
            f"({', '.join(c for c, _ in outlier_cols[:3])}) before training models."
        )

    if top_corr and abs(top_corr[0]["r"]) > 0.85:
        suggestions.append(
            f"'{top_corr[0]['a']}' and '{top_corr[0]['b']}' are highly correlated (r={top_corr[0]['r']}). "
            "Drop one to avoid multicollinearity in regression models."
        )

    if cat_cols:
        suggestions.append(
            f"Encode categorical columns ({', '.join(cat_cols[:3])}) using "
            "Label Encoding or One-Hot Encoding before applying ML algorithms."
        )

    if not suggestions:
        suggestions.append(
            "Dataset looks clean. Start with a train/test split and try a baseline model."
        )

    return {
        "insights":    insights[:4],
        "trends":      trends[:4],
        "anomalies":   anomalies[:4],
        "suggestions": suggestions[:4],
    }


def _answer_question(df: pd.DataFrame, payload: dict, q: str) -> str:
    """
    Rule-based Q&A: matches keywords in the question and returns
    a specific, data-driven answer using actual column names and values.
    """
    numeric_cols = payload.get("numeric_cols", [])
    cat_cols     = payload.get("cat_cols", [])
    stats        = payload.get("stats", {})
    top_corr     = payload.get("top_corr", [])
    missing_data = payload.get("missing_data", [])

    # ── correlation / relationship / affect / predict ──
    if any(w in q for w in ["correlat", "relationship", "affect", "predict", "depend", "impact", "influenc"]):
        if top_corr:
            top = top_corr[0]
            lines = [
                f"The strongest correlation in this dataset is between '{top['a']}' and '{top['b']}' "
                f"(r = {top['r']}), which is a {'strong' if abs(top['r']) > 0.7 else 'moderate'} "
                f"{'positive' if top['r'] > 0 else 'negative'} relationship."
            ]
            if len(top_corr) > 1:
                t2 = top_corr[1]
                lines.append(
                    f"The second strongest is '{t2['a']}' ↔ '{t2['b']}' (r = {t2['r']})."
                )
            return " ".join(lines)
        return "No strong correlations were found — the numeric columns appear largely independent."

    # ── missing / null / empty ──
    if any(w in q for w in ["missing", "null", "empty", "nan", "incomplete"]):
        if not missing_data:
            return "This dataset has no missing values — all columns are complete."
        total = sum(m["count"] for m in missing_data)
        worst = max(missing_data, key=lambda x: x["pct"])
        return (
            f"There are {total} missing values across {len(missing_data)} column(s). "
            f"The worst is '{worst['col']}' with {worst['count']} missing values ({worst['pct']}% of rows). "
            "Consider imputing with median/mode or dropping rows depending on the use case."
        )

    # ── outlier / anomaly ──
    if any(w in q for w in ["outlier", "anomal", "extreme", "unusual", "weird"]):
        results = []
        for col in numeric_cols:
            col_data = df[col].dropna()
            q1, q3 = col_data.quantile(0.25), col_data.quantile(0.75)
            iqr = q3 - q1
            n_out = int(((col_data < q1 - 1.5 * iqr) | (col_data > q3 + 1.5 * iqr)).sum())
            if n_out > 0:
                results.append((col, n_out))
        if not results:
            return "No outliers detected using the IQR method across any numeric column."
        results.sort(key=lambda x: x[1], reverse=True)
        summary = ", ".join(f"'{c}' ({n} outliers)" for c, n in results[:4])
        return f"Outliers detected (IQR method) in: {summary}. Use boxplots to visualise them."

    # ── average / mean / median ──
    if any(w in q for w in ["average", "mean", "median", "typical", "usual"]):
        # check if a specific column is mentioned
        for col in numeric_cols:
            if col.lower() in q:
                s = stats[col]
                return (
                    f"'{col}' has a mean of {s['mean']:,}, median of {s['p50']:,}, "
                    f"and standard deviation of {s['std']:,}. "
                    f"Values range from {s['min']:,} to {s['max']:,}."
                )
        # generic
        if stats:
            col = numeric_cols[0]
            s = stats[col]
            return (
                f"For example, '{col}' has mean = {s['mean']:,} and median = {s['p50']:,}. "
                f"Ask about a specific column for more detail."
            )

    # ── distribution / skew / spread ──
    if any(w in q for w in ["distribut", "skew", "spread", "range", "variab"]):
        if numeric_cols:
            skews = []
            for col in numeric_cols:
                skew = round(float(df[col].dropna().skew()), 2)
                skews.append((col, skew))
            skews.sort(key=lambda x: abs(x[1]), reverse=True)
            col, skew = skews[0]
            direction = "right-skewed (long tail toward higher values)" if skew > 0 else "left-skewed (long tail toward lower values)"
            return (
                f"'{col}' has the highest skewness ({skew}), meaning it is {direction}. "
                f"This often indicates the presence of outliers or a natural floor/ceiling effect."
            )

    # ── columns / features / variables ──
    if any(w in q for w in ["column", "feature", "variable", "field"]):
        return (
            f"This dataset has {df.shape[1]} columns: "
            f"{len(numeric_cols)} numeric ({', '.join(numeric_cols[:5])}) and "
            f"{len(cat_cols)} categorical ({', '.join(cat_cols[:5])})."
        )

    # ── rows / size / shape / how many ──
    if any(w in q for w in ["row", "size", "shape", "how many", "record", "sample", "count"]):
        return (
            f"The dataset has {df.shape[0]:,} rows and {df.shape[1]} columns. "
            f"After dropping rows with missing values, {int(df.dropna().shape[0]):,} complete rows remain."
        )

    # ── max / highest / largest / biggest ──
    if any(w in q for w in ["max", "highest", "largest", "biggest", "most", "top"]):
        for col in numeric_cols:
            if col.lower() in q:
                max_val = df[col].max()
                max_row = df[df[col] == max_val].iloc[0]
                return f"The maximum value in '{col}' is {max_val:,}."
        if numeric_cols:
            col = numeric_cols[0]
            return f"The highest value in '{col}' is {df[col].max():,} (mean: {stats[col]['mean']:,})."

    # ── min / lowest / smallest ──
    if any(w in q for w in ["min", "lowest", "smallest", "least"]):
        for col in numeric_cols:
            if col.lower() in q:
                return f"The minimum value in '{col}' is {df[col].min():,}."
        if numeric_cols:
            col = numeric_cols[0]
            return f"The lowest value in '{col}' is {df[col].min():,} (mean: {stats[col]['mean']:,})."

    # ── category / group / type ──
    if any(w in q for w in ["categor", "group", "type", "class", "label"]):
        if cat_cols:
            col = cat_cols[0]
            for c in cat_cols:
                if c.lower() in q:
                    col = c
                    break
            vc = df[col].value_counts()
            top3 = ", ".join(f"'{v}' ({n})" for v, n in vc.head(3).items())
            return (
                f"'{col}' has {df[col].nunique()} unique categories. "
                f"Top 3: {top3}."
            )

    # ── clean / quality / good ──
    if any(w in q for w in ["clean", "quality", "good", "ready", "usable"]):
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = int(df.isnull().sum().sum())
        pct_clean = round((1 - missing_cells / total_cells) * 100, 1)
        outlier_count = sum(
            int(((df[c].dropna() < df[c].quantile(0.25) - 1.5*(df[c].quantile(0.75)-df[c].quantile(0.25))) |
                 (df[c].dropna() > df[c].quantile(0.75) + 1.5*(df[c].quantile(0.75)-df[c].quantile(0.25)))).sum())
            for c in numeric_cols
        )
        return (
            f"The dataset is {pct_clean}% complete ({missing_cells} missing cells out of {total_cells:,}). "
            f"There are approximately {outlier_count} outlier values across all numeric columns. "
            f"{'It is in good shape for analysis.' if pct_clean > 95 else 'Some cleaning is recommended before modelling.'}"
        )

    # ── default fallback ──
    parts = []
    parts.append(f"This dataset has {df.shape[0]:,} rows and {df.shape[1]} columns.")
    if top_corr:
        t = top_corr[0]
        parts.append(f"The strongest relationship is '{t['a']}' ↔ '{t['b']}' (r = {t['r']}).")
    if missing_data:
        parts.append(f"There are {sum(m['count'] for m in missing_data)} missing values to address.")
    parts.append("Try asking about correlations, outliers, missing values, distributions, or specific columns.")
    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _get_df() -> pd.DataFrame:
    df = _store.get("df")
    if df is None:
        raise HTTPException(400, "No dataset loaded. Upload a CSV first.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FRONTEND
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("templates/index.html") as f:
        return f.read()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
