"""
AI-Powered Strategy Consulting Assistant — Streamlit UI

Launch:  streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

# Ensure the project root is on sys.path so `src.*` imports work
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.case_db import (
    build_all, load_index, retrieve, INDEX_PATH, META_PATH, get_model,
)
from src.predictor import predict
from src.explainer import explain

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Strategy Consulting Assistant",
    page_icon="🧠",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# Sidebar — system info & index management
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Strategy Assistant")
    st.markdown("---")

    st.markdown(
        "**How it works:**\n"
        "1. You describe a business problem\n"
        "2. The system encodes it with a sentence-transformer\n"
        "3. FAISS retrieves the most similar historical cases\n"
        "4. A weighted-vote predictor recommends a strategy\n"
        "5. The explainability layer shows *why*"
    )

    st.markdown("---")
    top_k = st.slider("Number of cases to retrieve", 1, 10, 5)

    st.markdown("### Optional context (improves explanations)")
    user_industry = st.selectbox(
        "Industry",
        ["(any)", "Retail", "Technology", "Healthcare", "Financial Services",
         "Manufacturing", "Energy", "Pharmaceuticals", "Automotive",
         "Airlines", "Telecommunications", "Food & Beverage",
         "Industrial Services", "Education", "Logistics",
         "Media & Entertainment", "Agriculture", "Government",
         "Construction", "Chemicals", "Mining", "Professional Services",
         "Consumer Goods"],
    )
    user_size = st.selectbox("Company size", ["(any)", "Small", "Medium", "Large"])

    st.markdown("---")
    if st.button("🔄 Rebuild index from cases.json"):
        with st.spinner("Building index …"):
            build_all()
        st.success("Index rebuilt ✓")


# ─────────────────────────────────────────────────────────────
# Load (or build) the index
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading case index …")
def load_or_build():
    if INDEX_PATH.exists() and META_PATH.exists():
        return load_index()
    return build_all()


index, cases = load_or_build()

# ─────────────────────────────────────────────────────────────
# Main panel
# ─────────────────────────────────────────────────────────────
st.header("AI-Powered Strategy Consulting Assistant")
st.caption(
    "Describe a business problem to find relevant historical cases and "
    "get an evidence-based strategy recommendation."
)

query = st.text_area(
    "Describe your business problem",
    height=150,
    placeholder=(
        "e.g. Our mid-sized SaaS company ($40M ARR) is seeing increasing churn "
        "among SMB customers due to aggressive pricing from newer competitors. "
        "We need a strategy to defend our market position while exploring "
        "upmarket enterprise expansion …"
    ),
)

if st.button("🔍 Analyse", type="primary") and query.strip():
    # ---- Retrieval ----
    with st.spinner("Embedding query & searching case database …"):
        results = retrieve(query, index, cases, top_k=top_k)

    # ---- Prediction ----
    rec = predict(results)

    # ---- Explainability ----
    query_features: dict = {}
    if user_industry != "(any)":
        query_features["industry"] = user_industry
    if user_size != "(any)":
        query_features["company_size"] = user_size
    expl = explain(query, query_features, results, rec)

    # ============================================================
    # Display: Recommendation summary
    # ============================================================
    st.markdown("---")
    st.subheader("📋 Recommendation")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recommended Strategy", rec.recommended_strategy)
    col2.metric("Predicted Outcome", rec.predicted_outcome,
                help=f"Confidence: {rec.outcome_confidence:.0%}")
    col3.metric("Est. ROI Range", f"{rec.estimated_roi_low}x – {rec.estimated_roi_high}x")
    col4.metric("Avg Time to Result", f"{rec.avg_time_to_result_months:.0f} months")

    st.info(rec.supporting_rationale)

    # ============================================================
    # Display: Confidence gauge
    # ============================================================
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rec.strategy_confidence * 100,
        title={"text": "Strategy Confidence"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2563eb"},
            "steps": [
                {"range": [0, 40], "color": "#fee2e2"},
                {"range": [40, 70], "color": "#fef3c7"},
                {"range": [70, 100], "color": "#d1fae5"},
            ],
        },
    ))
    fig_gauge.update_layout(height=250, margin=dict(t=40, b=20, l=40, r=40))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ============================================================
    # Display: Retrieved cases
    # ============================================================
    st.subheader("📚 Similar Historical Cases")

    for i, (case_expl) in enumerate(expl.case_explanations):
        case = next(c for c, _ in results if c.case_id == case_expl.case_id)
        score_pct = case_expl.similarity_score * 100

        with st.expander(
            f"**#{i+1} — {case.title}**  "
            f"({case.industry}, {case.year}) — "
            f"Similarity: {score_pct:.1f}%",
            expanded=(i == 0),
        ):
            # Top-level case info
            tags_col, meta_col = st.columns([3, 1])
            with tags_col:
                st.markdown(f"**Strategy:** {case.strategy_type}  |  "
                            f"**Outcome:** {case.outcome}  |  "
                            f"**ROI:** {case.outcome_metrics.roi_estimate}x  |  "
                            f"**Timeline:** {case.outcome_metrics.time_to_result_months} months")
                st.markdown(f"**Tags:** {', '.join(case.tags)}")
            with meta_col:
                st.markdown(f"**Company size:** {case.company_size}")
                st.markdown(f"**Region:** {case.region}")

            st.markdown("---")

            # Problem / Actions / Outcome
            st.markdown("**Problem:**")
            st.write(case.problem_description)
            st.markdown("**Key Challenges:**")
            for ch in case.key_challenges:
                st.markdown(f"- {ch}")
            st.markdown("**Actions Taken:**")
            st.write(case.actions_taken)
            st.markdown("**Outcome:**")
            st.write(case.outcome_description)

            # Explainability per case
            st.markdown("---")
            st.markdown("**🔍 Why this case was retrieved:**")

            if case_expl.key_overlap_phrases:
                st.markdown(
                    "Key overlapping concepts: "
                    + ", ".join(f"`{p}`" for p in case_expl.key_overlap_phrases)
                )
            if case_expl.shared_features:
                st.markdown(
                    "Matching features: "
                    + ", ".join(f"**{k}** = {v}" for k, v in case_expl.shared_features.items())
                )
            if case_expl.differing_features:
                st.markdown("Differences:")
                for feat, (qv, cv) in case_expl.differing_features.items():
                    st.markdown(f"- *{feat}*: your context = `{qv}`, case = `{cv}`")

    # ============================================================
    # Display: Similarity bar chart
    # ============================================================
    st.subheader("📊 Similarity Scores")
    labels = [f"{c.case_id}: {c.title[:40]}" for c, _ in results]
    scores_list = [round(s * 100, 1) for _, s in results]
    fig_bar = go.Figure(go.Bar(
        x=scores_list,
        y=labels,
        orientation="h",
        marker_color="#2563eb",
        text=[f"{s}%" for s in scores_list],
        textposition="auto",
    ))
    fig_bar.update_layout(
        xaxis_title="Cosine Similarity (%)",
        yaxis=dict(autorange="reversed"),
        height=40 + 50 * len(results),
        margin=dict(l=20, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ============================================================
    # Display: Strategy drivers
    # ============================================================
    st.subheader("⚙️ What's driving the recommendation")

    driver_cols = st.columns(len(expl.strategy_drivers))
    for col, d in zip(driver_cols, expl.strategy_drivers):
        icon = "✅" if d["direction"] == "supporting" else "🔄"
        col.markdown(
            f"{icon} **{d['factor']}**\n\n"
            f"Weight: {d['weight']:.0%}  ·  Cases: {d['case_count']}"
        )

    # ============================================================
    # Display: Outcome factor table
    # ============================================================
    st.subheader("📈 Outcome Evidence")
    for of in expl.outcome_factors:
        emoji = {"positive": "🟢", "cautionary": "🟡", "negative": "🔴"}[of["direction"]]
        st.markdown(
            f"{emoji} **{of['case']}** — {of['outcome']} "
            f"(ROI {of['roi']}x, weight {of['weight']:.2f})"
        )

    # ============================================================
    # Display: Confidence & methodology notes
    # ============================================================
    st.markdown("---")
    st.markdown(f"**Confidence Assessment:** {expl.confidence_note}")
    with st.expander("ℹ️ Methodology"):
        st.markdown(expl.methodology_note)

elif query.strip() == "" and st.session_state.get("_analyse_clicked"):
    st.warning("Please enter a business problem description.")
