"""
Enhanced AI Strategy Consulting Assistant with all improvements integrated.

New features:
- Hybrid retrieval with metadata filters
- Ensemble prediction
- Query caching
- Related cases & counter-examples
- Feedback collection
- Strategy comparison
- Analytics integration
"""

import sys
import uuid
from pathlib import Path
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# Ensure the project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.case_db import build_all, load_index, get_model, INDEX_PATH, META_PATH
from src.hybrid_retrieval import FilterConfig, retrieve_with_filters, get_filter_options
from src.ensemble_predictor import predict_ensemble, load_ensemble
from src.predictor import predict
from src.explainer import explain
from src.refined_retrieval import retrieve_related_cases, retrieve_counter_examples
from src.analytics import log_query, store_feedback, get_analytics_stats
from src.query_cache import get_cache
from src.llm_enhanced_explainer import generate_llm_explanation

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Strategy Consulting Assistant",
    page_icon="🧠",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# Load models & cache
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_or_build():
    if INDEX_PATH.exists() and META_PATH.exists():
        return load_index()
    return build_all()

@st.cache_resource
def load_ensemble_model():
    return load_ensemble()

@st.cache_resource
def get_query_cache():
    return get_cache()

index, cases = load_or_build()
ensemble = load_ensemble_model()
cache = get_query_cache()

# ─────────────────────────────────────────────────────────────
# Sidebar — filters & settings
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Strategy Assistant")
    st.markdown("**v2.0 — Now with AI enhancements**")
    st.markdown("---")

    st.markdown("### Retrieval Settings")
    top_k = st.slider("Number of cases to retrieve", 1, 10, 5)
    use_ensemble = st.checkbox("Use ensemble prediction", value=True, help="Combines k-NN, logistic regression, and random forest")
    use_llm_explanation = st.checkbox("Generate AI narrative", value=False, help="Uses Claude API for enhanced explanations (requires ANTHROPIC_API_KEY)")

    st.markdown("### Metadata Filters")
    st.caption("Narrow results by business context")

    industries = ["(any)"] + get_filter_options(cases)["industries"]
    user_industry = st.selectbox("Industry", industries)

    sizes = ["(any)"] + get_filter_options(cases)["company_sizes"]
    user_size = st.selectbox("Company size", sizes)

    outcomes = ["(any)"] + get_filter_options(cases)["outcomes"]
    user_outcome = st.selectbox("Outcome filter", outcomes, help="Filter by historical outcome")

    strategies = ["(any)"] + get_filter_options(cases)["strategy_types"]
    user_strategy = st.selectbox("Strategy type", strategies, help="Filter by strategy used")

    st.markdown("---")
    st.markdown("### Cache Statistics")
    cache_stats = cache.get_stats()
    st.metric("Cached Queries", cache_stats["cached_queries"])
    st.metric("Cache Hits", cache_stats["total_accesses"])

    st.markdown("---")
    if st.button("🔄 Rebuild index"):
        with st.spinner("Rebuilding..."):
            build_all()
            cache.invalidate_all()
        st.success("Index rebuilt ✓")
        st.rerun()

# ─────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────
st.header("AI-Powered Strategy Consulting Assistant")
st.caption(
    "Describe your business problem to get evidence-based recommendations from "
    f"{len(cases)} historical cases. Now with hybrid search, ensemble AI, and caching."
)

# Tabs for different views
tab_consult, tab_comparison, tab_analytics = st.tabs(["💼 Consultation", "📊 Strategy Comparison", "📈 Analytics"])

# ─────────────────────────────────────────────────────────────
# TAB 1: Consultation
# ─────────────────────────────────────────────────────────────
with tab_consult:
    query = st.text_area(
        "Describe your business problem",
        height=150,
        placeholder=(
            "e.g. Our mid-sized SaaS company ($40M ARR) is seeing increasing churn "
            "among SMB customers due to aggressive pricing from newer competitors. "
            "We need a strategy to defend our market position while exploring "
            "upmarket enterprise expansion..."
        ),
        key="query_input"
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        submit = st.button("🔍 Get Recommendation", type="primary", use_container_width=True)
    with col2:
        show_related = st.checkbox("Show related cases", value=False)
    with col3:
        show_counter = st.checkbox("Show alternatives", value=False)

    if submit and query:
        query_id = str(uuid.uuid4())

        with st.spinner("Analyzing your problem..."):
            # Build filters
            filters = FilterConfig(
                industry=None if user_industry == "(any)" else user_industry,
                company_size=None if user_size == "(any)" else user_size,
                outcome=None if user_outcome == "(any)" else user_outcome,
                strategy_type=None if user_strategy == "(any)" else user_strategy,
            )

            # Check cache first
            cache_key = {
                "industry": filters.industry,
                "company_size": filters.company_size,
                "outcome": filters.outcome,
                "strategy_type": filters.strategy_type,
            }
            cached = cache.get(query, cache_key)

            if cached:
                st.info("⚡ Retrieved from cache")
                embedding, recommendation, retrieved_case_ids = cached
                retrieved = [(c, 0.95) for c in cases if c.case_id in retrieved_case_ids]
            else:
                # Embed query
                model = get_model()
                q_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype('float32')

                # Hybrid retrieval
                retrieved = retrieve_with_filters(query, index, cases, filters, top_k)

                # Predict
                if use_ensemble and ensemble:
                    recommendation = predict_ensemble(retrieved, ensemble)
                else:
                    recommendation = predict(retrieved)

                # Cache result
                cache.set(query, q_emb[0], recommendation, [c.case_id for c, _ in retrieved], cache_key)

            # Log query
            query_features = {"industry": filters.industry, "company_size": filters.company_size}
            log_query(query_id, query, retrieved, recommendation, filters.industry, filters.company_size, top_k)

        # ─── Display recommendation ───
        st.markdown("---")
        st.subheader("📋 Recommendation")

        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown(f"### {recommendation.recommended_strategy}")
            st.markdown(f"**Confidence:** {recommendation.strategy_confidence:.0%}")
            st.markdown(f"**Predicted Outcome:** {recommendation.predicted_outcome}")

            st.markdown("#### Key Metrics")
            st.markdown(f"- **Estimated ROI:** {recommendation.estimated_roi_low:.1f}x – {recommendation.estimated_roi_high:.1f}x")
            st.markdown(f"- **Time to Results:** ~{recommendation.avg_time_to_result_months:.0f} months")

            st.markdown("#### Rationale")
            st.info(recommendation.supporting_rationale)

            # LLM-enhanced explanation
            if use_llm_explanation:
                with st.expander("📝 AI-Generated Narrative Explanation", expanded=True):
                    with st.spinner("Generating explanation..."):
                        llm_explanation = generate_llm_explanation(query, retrieved, recommendation, max_tokens=600)
                    st.markdown(llm_explanation)

        with col_right:
            # Confidence gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=recommendation.strategy_confidence * 100,
                title={'text': "Confidence"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 75], 'color': "gray"},
                        {'range': [75, 100], 'color': "lightblue"}
                    ],
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)

        # ─── Retrieved cases ───
        st.markdown("---")
        st.subheader(f"📚 {len(retrieved)} Similar Historical Cases")

        for i, (case, score) in enumerate(retrieved, 1):
            with st.expander(f"#{i} — {case.title} ({case.industry}) — Similarity: {score:.2%}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Strategy Used:** {case.strategy_type}")
                    st.markdown(f"**Outcome:** {case.outcome}")
                    st.markdown(f"**ROI:** {case.outcome_metrics.roi_estimate:.1f}x")
                    st.markdown(f"**Timeline:** {case.outcome_metrics.time_to_result_months} months")

                with col2:
                    st.markdown(f"**Company Size:** {case.company_size}")
                    st.markdown(f"**Region:** {case.region}")
                    st.markdown(f"**Year:** {case.year}")

                st.markdown("**Problem Description:**")
                st.caption(case.problem_description)

                st.markdown("**Key Challenges:**")
                st.caption(", ".join(case.key_challenges))

                st.markdown("**Actions Taken:**")
                st.caption(case.actions_taken)

                st.markdown("**Outcome:**")
                st.caption(case.outcome_description)

        # ─── Related cases ───
        if show_related and retrieved:
            st.markdown("---")
            st.subheader("🔗 Related Cases")
            st.caption("Cases similar to your top match")

            related = retrieve_related_cases(retrieved[0][0].case_id, cases, index, top_k=3)

            for case, score in related:
                with st.expander(f"{case.title} ({case.strategy_type}) — {score:.2%}"):
                    st.markdown(f"**Outcome:** {case.outcome} | **ROI:** {case.outcome_metrics.roi_estimate:.1f}x")
                    st.caption(case.problem_description)

        # ─── Counter-examples ───
        if show_counter:
            st.markdown("---")
            st.subheader("🔄 Alternative Strategies That Also Worked")
            st.caption("Different approaches with positive outcomes")

            counter = retrieve_counter_examples(recommendation.recommended_strategy, cases, index, top_k=3)

            if counter:
                for case, _ in counter:
                    with st.expander(f"{case.title} — {case.strategy_type}"):
                        st.markdown(f"**ROI:** {case.outcome_metrics.roi_estimate:.1f}x | **Timeline:** {case.outcome_metrics.time_to_result_months} months")
                        st.caption(case.problem_description)
            else:
                st.info("No alternative successful strategies found in database.")

        # ─── Feedback ───
        st.markdown("---")
        st.subheader("💬 Was this recommendation helpful?")

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            rating = st.slider("Rating (1-5)", 1, 5, 3, key=f"rating_{query_id}")

        with col2:
            helpful = st.radio("Helpful?", ["Yes", "No"], key=f"helpful_{query_id}")

        with col3:
            feedback_notes = st.text_input("Additional notes (optional)", key=f"notes_{query_id}")

        if st.button("Submit Feedback"):
            store_feedback(query_id, rating, helpful == "Yes", feedback_notes or None)
            st.success("Thank you for your feedback!")

# ─────────────────────────────────────────────────────────────
# TAB 2: Strategy Comparison
# ─────────────────────────────────────────────────────────────
with tab_comparison:
    st.subheader("📊 Strategy Performance Comparison")
    st.caption("Compare strategy types across key metrics")

    # Compute strategy statistics
    strategy_stats = {}
    for case in cases:
        if case.strategy_type not in strategy_stats:
            strategy_stats[case.strategy_type] = {
                "positive": 0,
                "mixed": 0,
                "negative": 0,
                "rois": [],
                "times": [],
                "total": 0
            }

        stats = strategy_stats[case.strategy_type]
        stats["total"] += 1
        stats["rois"].append(case.outcome_metrics.roi_estimate)
        stats["times"].append(case.outcome_metrics.time_to_result_months)

        if case.outcome == "Positive":
            stats["positive"] += 1
        elif case.outcome == "Mixed":
            stats["mixed"] += 1
        else:
            stats["negative"] += 1

    # Create comparison DataFrame
    comparison_data = []
    for strategy, stats in strategy_stats.items():
        comparison_data.append({
            "Strategy": strategy,
            "Cases": stats["total"],
            "Success Rate": f"{stats['positive']/stats['total']:.0%}",
            "Avg ROI": f"{sum(stats['rois'])/len(stats['rois']):.1f}x",
            "Avg Timeline": f"{sum(stats['times'])/len(stats['times']):.0f} mo",
            "Positive": stats["positive"],
            "Mixed": stats["mixed"],
            "Negative": stats["negative"],
        })

    df = pd.DataFrame(comparison_data).sort_values("Cases", ascending=False)

    st.dataframe(df, use_container_width=True, hide_index=True)

    # Visualization
    col1, col2 = st.columns(2)

    with col1:
        # Success rate chart
        fig = go.Figure(data=[
            go.Bar(
                x=[row["Strategy"] for _, row in df.iterrows()],
                y=[int(row["Success Rate"].strip('%')) for _, row in df.iterrows()],
                marker_color='lightblue'
            )
        ])
        fig.update_layout(title="Success Rate by Strategy", yaxis_title="Success Rate (%)", xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # ROI comparison
        fig = go.Figure(data=[
            go.Bar(
                x=[row["Strategy"] for _, row in df.iterrows()],
                y=[float(row["Avg ROI"].replace('x', '')) for _, row in df.iterrows()],
                marker_color='lightgreen'
            )
        ])
        fig.update_layout(title="Average ROI by Strategy", yaxis_title="ROI (multiplier)", xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# TAB 3: Analytics
# ─────────────────────────────────────────────────────────────
with tab_analytics:
    st.subheader("📈 Usage Analytics")

    stats = get_analytics_stats()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Queries", stats["total_queries"])

    with col2:
        st.metric("Avg Confidence", f"{stats['avg_confidence']:.1%}")

    with col3:
        st.metric("Feedback Received", stats["feedback_count"])

    with col4:
        if stats["feedback_count"] > 0:
            st.metric("Avg User Rating", f"{stats['avg_rating']:.1f}/5")
        else:
            st.metric("Avg User Rating", "N/A")

    if stats["total_queries"] > 0:
        st.markdown("---")
        st.markdown("### Top Strategies Recommended")

        top_strategies = stats["top_strategies"]
        if top_strategies:
            for strategy, count in top_strategies:
                st.markdown(f"- **{strategy}**: {count} times")

        st.markdown("---")
        st.markdown("### Industry Distribution")

        industry_dist = stats["industry_distribution"]
        if industry_dist:
            fig = go.Figure(data=[go.Pie(
                labels=list(industry_dist.keys()),
                values=list(industry_dist.values())
            )])
            fig.update_layout(title="Queries by Industry")
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No query data yet. Start by submitting a consultation!")

st.markdown("---")
st.caption(f"AI Strategy Consulting Assistant v2.0 | {len(cases)} cases loaded | Powered by FAISS + ensemble AI")
