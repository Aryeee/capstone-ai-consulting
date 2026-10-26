"""StrategyOS — Streamlit consultation workspace.

Launch with: streamlit run app.py
"""

import sys
import time
import uuid
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics import get_analytics_stats, log_query, store_feedback
from src.case_db import INDEX_PATH, META_PATH, build_all, load_index
from src.ensemble_predictor import load_ensemble, predict_ensemble
from src.explainer import explain
from src.hybrid_retrieval import FilterConfig, get_filter_options, retrieve_with_filters
from src.predictor import predict
from src.refined_retrieval import retrieve_counter_examples, retrieve_related_cases

st.set_page_config(page_title="StrategyOS — AI Consulting Workspace", page_icon="✦", layout="wide")

# StrategyOS visual system: a quiet, editorial workspace instead of the default
# Streamlit chrome. The same palette is used by the Vercel dashboard.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,600;1,600&display=swap');
    :root { --ink:#17221e; --muted:#718078; --line:#dfe8e2; --accent:#0f9f78; --accent-dark:#086b52; --soft:#eef5f0; --paper:#f7f9f7; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
    body { background:var(--paper); }
    #MainMenu, footer, [data-testid="stSidebar"], [data-testid="collapsedControl"] { display:none!important; }
    .block-container { max-width:1240px; padding:0 2.6rem 4rem; }
    .topbar { display:flex; justify-content:space-between; align-items:center; padding:1.25rem 0 1.1rem; border-bottom:1px solid var(--line); margin-bottom:2.8rem; }
    .brand { display:flex; align-items:center; gap:.7rem; }
    .brand-mark { display:grid; place-items:center; width:32px; height:32px; border-radius:9px; color:#153d31; background:#a6e2c9; font-size:19px; }
    .brand-name { margin:0; font-weight:700; letter-spacing:-.04em; font-size:1.08rem; }.brand-name span { color:var(--accent); font-weight:500; }
    .brand-sub { margin:2px 0 0; color:var(--muted); font:9px 'DM Mono',monospace; letter-spacing:.4px; text-transform:uppercase; }
    .top-status { display:flex; align-items:center; gap:.45rem; color:var(--muted); font:10px 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.5px; }
    .status-dot { width:6px; height:6px; border-radius:50%; background:#53bd8e; box-shadow:0 0 0 4px #e0f4e8; }
    h1,h2,h3,h4 { letter-spacing:-.04em; } h1 { font-size:2.8rem!important; }
    .eyebrow { color:var(--accent); font:10px 'DM Mono',monospace; letter-spacing:1.1px; text-transform:uppercase; }
    .hero { padding:0 0 2rem; border-bottom:1px solid var(--line); margin-bottom:2rem; }.hero h1 em { color:var(--accent); font-family:'Playfair Display',serif; }.hero p { max-width:650px; color:var(--muted); line-height:1.65; }
    .card { border:1px solid var(--line); border-radius:13px; padding:1.35rem; background:#fff; margin:0 0 1rem; box-shadow:0 8px 30px rgba(23,34,30,.045); }.recommendation { border-left:4px solid var(--accent); background:#f7fcf9; }.recommendation h2 { margin:.3rem 0 .7rem; }.muted { color:var(--muted); font-size:.85rem; line-height:1.6; }
    [data-testid="stTabs"] { margin-top:1.4rem; }.stTabs [data-baseweb="tab-list"] { gap:1.55rem; border-bottom:1px solid var(--line); }.stTabs [data-baseweb="tab"] { color:var(--muted); font-size:.82rem; padding:0 0 12px; }.stTabs [aria-selected="true"] { color:var(--accent-dark); font-weight:700; }.stTabs [data-baseweb="tab-highlight"] { background:var(--accent); height:2px; }
    [data-testid="stTextArea"] textarea, [data-baseweb="select"] > div { border-color:var(--line); border-radius:8px; background:#fff; }.stTextArea textarea:focus { border-color:#88c9ae; box-shadow:0 0 0 3px rgba(15,159,120,.08); }
    [data-testid="stSelectbox"] [data-baseweb="select"] * { color:var(--ink)!important; }
    [data-testid="stSelectbox"] [data-baseweb="select"] input { color:var(--ink)!important; -webkit-text-fill-color:var(--ink)!important; }
    [data-baseweb="popover"] [role="listbox"], [data-baseweb="popover"] [role="option"] { background:#fff!important; color:var(--ink)!important; }
    [data-baseweb="popover"] [role="option"] * { color:var(--ink)!important; }
    [data-baseweb="popover"] [role="option"]:hover, [data-baseweb="popover"] [role="option"][aria-selected="true"] { background:var(--soft)!important; color:var(--accent-dark)!important; }
    label p { color:var(--muted)!important; font-size:.75rem!important; }.stSlider [data-baseweb="slider"] { margin-top:.2rem; }
    div.stButton > button { border-radius:8px; border:1px solid var(--line); min-height:2.45rem; font-weight:600; }.stButton > button[kind="primary"] { background:#0f9f78; border-color:#0f9f78; color:#fff; box-shadow:0 5px 14px rgba(15,159,120,.18); }.stButton > button[kind="primary"]:hover { background:#086b52; border-color:#086b52; }
    [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:9px; padding:.75rem 1rem; }.stMetric label { color:var(--muted); font:10px 'DM Mono',monospace; text-transform:uppercase; }.stMetric [data-testid="stMetricValue"] { font-size:1.15rem; }
    [data-testid="stExpander"] { border:1px solid var(--line); border-radius:9px; background:#fff; margin:.55rem 0; }.streamlit-expanderHeader { font-size:.84rem; }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; overflow:hidden; }.status { font:10px 'DM Mono',monospace; color:#0f9f78; }
    @media (max-width:700px) { .block-container { padding:0 1rem 3rem; }.topbar { margin-bottom:2rem; }.top-status { display:none; } h1 { font-size:2.25rem!important; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading case library…")
def load_or_build():
    if INDEX_PATH.exists() and META_PATH.exists():
        return load_index()
    return build_all()


@st.cache_resource(show_spinner="Loading ensemble models…")
def load_ensemble_cached():
    return load_ensemble()


index, cases = load_or_build()
ensemble = load_ensemble_cached()
options = get_filter_options(cases)

st.markdown('<div class="topbar"><div class="brand"><div class="brand-mark">✦</div><div><p class="brand-name">Strategy<span>OS</span></p><p class="brand-sub">AI consulting workspace</p></div></div><div class="top-status"><span class="status-dot"></span>Local & private · ' + str(len(cases)) + ' cases</div></div>', unsafe_allow_html=True)

with st.expander("Workspace controls", expanded=False):
    control_left, control_right = st.columns([3, 1])
    with control_left:
        st.caption("Your query is embedded and matched against the local case library. No business context is sent to a remote service.")
    with control_right:
        if st.button("Rebuild case library", use_container_width=True):
            with st.spinner("Rebuilding index…"):
                build_all()
                st.cache_resource.clear()
            st.success("Case library rebuilt.")
            st.rerun()

st.markdown('<div class="hero"><div class="eyebrow">CASE-BASED REASONING</div><h1>Turn a complex problem<br><em>into a clear next move.</em></h1><p>Describe the business challenge. StrategyOS finds analogous cases, weighs their outcomes, and makes the reasoning visible.</p></div>', unsafe_allow_html=True)

consult_tab, library_tab, analytics_tab = st.tabs(["Consultation", "Strategy library", "Analytics"])

with consult_tab:
    left, right = st.columns([1.65, .8])
    with left:
        st.markdown('<div class="eyebrow">01 / PROBLEM BRIEF</div>', unsafe_allow_html=True)
        st.subheader("What are you solving?")
        query = st.text_area(
            "Business problem",
            height=180,
            label_visibility="collapsed",
            placeholder="e.g. Our mid-sized SaaS company is seeing increasing churn among SMB customers due to aggressive pricing from newer competitors. We need to defend our position while exploring enterprise expansion…",
        )
    with right:
        st.markdown('<div class="eyebrow">02 / CONTEXT</div>', unsafe_allow_html=True)
        st.subheader("Refine the lens")
        top_k = st.slider("Cases to retrieve", 1, 10, 5)
        industry = st.selectbox("Industry", ["(any)"] + options["industries"])
        company_size = st.selectbox("Company size", ["(any)"] + options["company_sizes"])
        outcome = st.selectbox("Historical outcome", ["(any)"] + options["outcomes"])
        strategy = st.selectbox("Strategy type", ["(any)"] + options["strategy_types"])

    related_col, alternatives_col, ensemble_col = st.columns(3)
    with related_col:
        show_related = st.checkbox("Show related cases")
    with alternatives_col:
        show_alternatives = st.checkbox("Show alternatives")
    with ensemble_col:
        use_ensemble = st.checkbox("Use ensemble prediction", value=ensemble is not None)

    analyze = st.button("Analyse problem  →", type="primary", use_container_width=True)
    if analyze:
        if len(query.strip()) < 10:
            st.warning("Please provide at least 10 characters describing the problem.")
        else:
            started = time.perf_counter()
            query_id = str(uuid.uuid4())
            filters = FilterConfig(
                industry=None if industry == "(any)" else industry,
                company_size=None if company_size == "(any)" else company_size,
                outcome=None if outcome == "(any)" else outcome,
                strategy_type=None if strategy == "(any)" else strategy,
            )
            with st.spinner("Finding analogous cases and evaluating outcomes…"):
                retrieved = retrieve_with_filters(query, index, cases, filters, top_k)
                recommendation = predict_ensemble(retrieved, ensemble) if use_ensemble and ensemble else predict(retrieved)
                features = {k: v for k, v in {"industry": filters.industry, "company_size": filters.company_size}.items() if v}
                explanation = explain(query, features, retrieved, recommendation)
                log_query(query_id, query, retrieved, recommendation, filters.industry, filters.company_size, top_k)
            elapsed = round((time.perf_counter() - started) * 1000)
            st.session_state["analysis"] = {
                "query_id": query_id, "query": query, "retrieved": retrieved,
                "recommendation": recommendation, "explanation": explanation,
                "elapsed": elapsed, "filters": filters,
                "show_related": show_related, "show_alternatives": show_alternatives,
                "index": index, "cases": cases,
            }

    analysis = st.session_state.get("analysis")
    if analysis:
        retrieved = analysis["retrieved"]
        rec = analysis["recommendation"]
        expl = analysis["explanation"]
        st.markdown("---")
        st.markdown('<div class="eyebrow">03 / RECOMMENDATION</div>', unsafe_allow_html=True)
        st.subheader("Your evidence-backed direction")
        rec_left, rec_right = st.columns([1.7, .7])
        with rec_left:
            st.markdown(f'<div class="card recommendation"><div class="eyebrow">RECOMMENDED STRATEGY</div><h2>{rec.recommended_strategy}</h2><p class="muted">{rec.supporting_rationale}</p></div>', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted outcome", rec.predicted_outcome, f"{rec.outcome_confidence:.0%} confidence")
            m2.metric("Estimated ROI", f"{rec.estimated_roi_low}x – {rec.estimated_roi_high}x")
            m3.metric("Time to result", f"{rec.avg_time_to_result_months:.0f} months")
        with rec_right:
            gauge = go.Figure(go.Indicator(mode="gauge+number", value=rec.strategy_confidence * 100, title={"text": "Strategy confidence"}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#0f9f78"}, "steps": [{"range": [0, 40], "color": "#fbe7e5"}, {"range": [40, 70], "color": "#fdf1d8"}, {"range": [70, 100], "color": "#e5f6ed"}]}))
            gauge.update_layout(height=230, margin=dict(t=35, b=10, l=20, r=20))
            st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
        st.caption(f"{len(retrieved)} cases · {analysis['elapsed']}ms")

        st.markdown('<div class="card"><div class="eyebrow">EVIDENCE MAP</div><h3>Why these cases surfaced</h3></div>', unsafe_allow_html=True)
        sim_df = pd.DataFrame({"Case": [c.title for c, _ in retrieved], "Similarity": [round(score * 100, 1) for _, score in retrieved]})
        fig = go.Figure(go.Bar(x=sim_df["Similarity"], y=sim_df["Case"], orientation="h", marker_color="#0f9f78", text=sim_df["Similarity"].map(lambda x: f"{x}%"), textposition="auto"))
        fig.update_layout(height=max(220, 55 * len(retrieved)), margin=dict(l=10, r=20, t=10, b=20), yaxis={"autorange": "reversed"}, xaxis_title="Similarity")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="eyebrow">HISTORICAL EVIDENCE</div>', unsafe_allow_html=True)
        st.subheader("Similar cases")
        for i, (case, score) in enumerate(retrieved):
            with st.expander(f"{i + 1:02d}  {case.title} · {case.industry} · {case.year} · {score:.1%} similar", expanded=i == 0):
                a, b, c = st.columns(3)
                a.metric("Strategy", case.strategy_type)
                b.metric("ROI", f"{case.outcome_metrics.roi_estimate}x")
                c.metric("Outcome", case.outcome)
                st.markdown(f"**Problem**  \n{case.problem_description}")
                st.markdown(f"**Actions taken**  \n{case.actions_taken}")
                st.markdown(f"**Outcome**  \n{case.outcome_description}")
                case_expl = expl.case_explanations[i]
                if case_expl.key_overlap_phrases:
                    st.caption("Shared concepts: " + ", ".join(case_expl.key_overlap_phrases))
                if case_expl.shared_features:
                    st.caption("Matching context: " + ", ".join(f"{k}={v}" for k, v in case_expl.shared_features.items()))

        if analysis["show_related"] and retrieved:
            st.subheader("Related cases")
            for case, score in retrieve_related_cases(retrieved[0][0].case_id, cases, index, top_k=3):
                st.markdown(f"- **{case.title}** — {case.strategy_type} · {score:.1%} similar · ROI {case.outcome_metrics.roi_estimate}x")
        if analysis["show_alternatives"]:
            st.subheader("Alternative strategies that also worked")
            for case, _ in retrieve_counter_examples(rec.recommended_strategy, cases, index, top_k=3):
                st.markdown(f"- **{case.title}** — {case.strategy_type} · ROI {case.outcome_metrics.roi_estimate}x")

        st.markdown("---")
        st.subheader("Close the loop")
        rating = st.radio("Was this direction useful?", [1, 2, 3, 4, 5], horizontal=True, key=f"rating_{analysis['query_id']}")
        if st.button("Submit feedback", key=f"feedback_{analysis['query_id']}"):
            store_feedback(analysis["query_id"], rating, rating >= 4)
            st.success("Thanks — your feedback was saved.")
        with st.expander("Methodology and limitations"):
            st.write(expl.methodology_note)
    else:
        st.info("Start with a clear problem brief. The recommendation and evidence map will appear here.")

with library_tab:
    st.markdown('<div class="eyebrow">CASE LIBRARY</div>', unsafe_allow_html=True)
    st.header("Compare strategy patterns")
    stats = []
    for strategy_name in options["strategy_types"]:
        group = [c for c in cases if c.strategy_type == strategy_name]
        positive = sum(c.outcome == "Positive" for c in group)
        stats.append({"Strategy": strategy_name, "Cases": len(group), "Success rate": positive / len(group), "Avg ROI": sum(c.outcome_metrics.roi_estimate for c in group) / len(group), "Avg timeline": sum(c.outcome_metrics.time_to_result_months for c in group) / len(group), "Positive": positive, "Mixed": sum(c.outcome == "Mixed" for c in group), "Negative": sum(c.outcome == "Negative" for c in group)})
    library_df = pd.DataFrame(stats).sort_values("Success rate", ascending=False)
    st.dataframe(library_df.style.format({"Success rate": "{:.0%}", "Avg ROI": "{:.1f}x", "Avg timeline": "{:.0f} mo"}), use_container_width=True, hide_index=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(go.Figure(go.Bar(x=library_df["Strategy"], y=library_df["Success rate"] * 100, marker_color="#0f9f78"), layout={"title": "Success rate by strategy", "yaxis_title": "%", "xaxis_tickangle": -35}), use_container_width=True, config={"displayModeBar": False})
    with col2:
        st.plotly_chart(go.Figure(go.Bar(x=library_df["Strategy"], y=library_df["Avg ROI"], marker_color="#d7a54a"), layout={"title": "Average ROI by strategy", "yaxis_title": "Multiplier", "xaxis_tickangle": -35}), use_container_width=True, config={"displayModeBar": False})

with analytics_tab:
    st.markdown('<div class="eyebrow">LOCAL TELEMETRY</div>', unsafe_allow_html=True)
    st.header("Workspace analytics")
    stats = get_analytics_stats()
    a, b, c, d = st.columns(4)
    a.metric("Total consultations", stats["total_queries"])
    b.metric("Average confidence", f'{stats["avg_confidence"]:.0%}')
    c.metric("Feedback received", stats["feedback_count"])
    d.metric("Average rating", f'{stats["avg_rating"]:.1f}/5' if stats["feedback_count"] else "—")
    if stats["top_strategies"]:
        st.subheader("Recommendation mix")
        st.bar_chart(pd.DataFrame(stats["top_strategies"], columns=["Strategy", "Count"]).set_index("Strategy"))
        st.subheader("Industry distribution")
        if stats["industry_distribution"]:
            st.dataframe(pd.DataFrame(list(stats["industry_distribution"].items()), columns=["Industry", "Queries"]), use_container_width=True, hide_index=True)
    else:
        st.info("Run your first consultation to build the analytics view.")

st.caption(f"StrategyOS v2.0 · {len(cases)} cases loaded · local case-based reasoning")
