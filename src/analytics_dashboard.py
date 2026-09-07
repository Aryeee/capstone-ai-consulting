"""
Analytics Dashboard — Track usage, popular queries, and performance metrics.

Run with: streamlit run src/analytics_dashboard.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import json
from pathlib import Path
from collections import Counter

# ─────────────────────────────────────────────────────────────
# Mock Analytics Data (In production, use a database)
# ─────────────────────────────────────────────────────────────

ANALYTICS_PATH = Path(__file__).resolve().parent.parent / "data" / "analytics.json"


def load_analytics():
    """Load analytics data from JSON file."""
    if not ANALYTICS_PATH.exists():
        return {
            "queries": [],
            "daily_stats": [],
            "strategy_distribution": {},
            "industry_distribution": {},
        }
    with open(ANALYTICS_PATH, "r") as f:
        return json.load(f)


def save_analytics(data):
    """Save analytics data to JSON file."""
    ANALYTICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ANALYTICS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def log_query(query_text, recommended_strategy, confidence, response_time_ms):
    """Log a query for analytics."""
    data = load_analytics()

    data["queries"].append({
        "timestamp": datetime.now().isoformat(),
        "query": query_text[:100],  # Store first 100 chars
        "strategy": recommended_strategy,
        "confidence": confidence,
        "response_time_ms": response_time_ms,
    })

    # Update distributions
    data["strategy_distribution"][recommended_strategy] = (
        data["strategy_distribution"].get(recommended_strategy, 0) + 1
    )

    # Keep only last 1000 queries
    data["queries"] = data["queries"][-1000:]

    save_analytics(data)


def generate_mock_data():
    """Generate mock analytics data for demonstration."""
    strategies = [
        "Digital Transformation",
        "Cost Optimization",
        "Market Expansion",
        "Operational Excellence",
        "Customer Experience",
        "Innovation Strategy",
    ]

    industries = [
        "Retail", "Technology", "Healthcare", "Financial Services",
        "Manufacturing", "Energy", "Telecommunications"
    ]

    queries = []
    for i in range(100):
        days_ago = i % 30
        timestamp = (datetime.now() - timedelta(days=days_ago)).isoformat()
        queries.append({
            "timestamp": timestamp,
            "query": f"Sample query {i}",
            "strategy": strategies[i % len(strategies)],
            "confidence": 0.6 + (i % 4) * 0.1,
            "response_time_ms": 500 + (i % 10) * 100,
            "industry": industries[i % len(industries)],
        })

    return {
        "queries": queries,
        "strategy_distribution": dict(Counter([q["strategy"] for q in queries])),
        "industry_distribution": dict(Counter([q["industry"] for q in queries])),
    }


# ─────────────────────────────────────────────────────────────
# Dashboard UI
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AI Strategy Consulting Assistant — Analytics Dashboard")
st.caption("Real-time insights into usage patterns and performance")

# Load data
if st.sidebar.button("🔄 Generate Mock Data"):
    save_analytics(generate_mock_data())
    st.rerun()

data = load_analytics()

if not data["queries"]:
    st.warning("No analytics data available. Click 'Generate Mock Data' in the sidebar to see a demo.")
    st.stop()

# Convert to DataFrame
df = pd.DataFrame(data["queries"])
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = df["timestamp"].dt.date

# ─────────────────────────────────────────────────────────────
# Key Metrics
# ─────────────────────────────────────────────────────────────

st.header("Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Queries", len(df))

with col2:
    avg_confidence = df["confidence"].mean()
    st.metric("Avg Confidence", f"{avg_confidence:.2%}")

with col3:
    avg_response = df["response_time_ms"].mean()
    st.metric("Avg Response Time", f"{avg_response:.0f}ms")

with col4:
    recent_queries = df[df["timestamp"] > datetime.now() - timedelta(days=7)]
    st.metric("Queries (7d)", len(recent_queries))

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📈 Queries Over Time")

    daily_counts = df.groupby("date").size().reset_index(name="count")
    fig_timeline = px.line(
        daily_counts,
        x="date",
        y="count",
        markers=True,
        title="Daily Query Volume",
    )
    fig_timeline.update_layout(height=300, margin=dict(t=40, b=40, l=40, r=40))
    st.plotly_chart(fig_timeline, use_container_width=True)

with col_right:
    st.subheader("⚡ Response Time Distribution")

    fig_response = px.histogram(
        df,
        x="response_time_ms",
        nbins=20,
        title="Response Time Histogram",
    )
    fig_response.update_layout(height=300, margin=dict(t=40, b=40, l=40, r=40))
    st.plotly_chart(fig_response, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Strategy Distribution
# ─────────────────────────────────────────────────────────────

st.subheader("🎯 Recommended Strategies")

col1, col2 = st.columns(2)

with col1:
    strat_df = pd.DataFrame(
        data["strategy_distribution"].items(),
        columns=["Strategy", "Count"]
    ).sort_values("Count", ascending=False)

    fig_strat_bar = px.bar(
        strat_df,
        x="Strategy",
        y="Count",
        title="Strategy Recommendation Frequency",
        color="Count",
        color_continuous_scale="Blues",
    )
    fig_strat_bar.update_layout(height=350, margin=dict(t=40, b=40, l=40, r=40))
    st.plotly_chart(fig_strat_bar, use_container_width=True)

with col2:
    fig_strat_pie = px.pie(
        strat_df,
        names="Strategy",
        values="Count",
        title="Strategy Distribution",
    )
    fig_strat_pie.update_layout(height=350, margin=dict(t=40, b=40, l=40, r=40))
    st.plotly_chart(fig_strat_pie, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Industry Distribution (if available)
# ─────────────────────────────────────────────────────────────

if "industry" in df.columns:
    st.subheader("🏢 Industry Distribution")

    industry_counts = df["industry"].value_counts().reset_index()
    industry_counts.columns = ["Industry", "Count"]

    fig_industry = px.bar(
        industry_counts,
        x="Industry",
        y="Count",
        title="Queries by Industry",
        color="Count",
        color_continuous_scale="Viridis",
    )
    fig_industry.update_layout(height=300, margin=dict(t=40, b=40, l=40, r=40))
    st.plotly_chart(fig_industry, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Confidence Analysis
# ─────────────────────────────────────────────────────────────

st.subheader("💯 Confidence Analysis")

col1, col2 = st.columns(2)

with col1:
    fig_conf_hist = px.histogram(
        df,
        x="confidence",
        nbins=20,
        title="Confidence Score Distribution",
    )
    fig_conf_hist.update_layout(height=300, margin=dict(t=40, b=40, l=40, r=40))
    st.plotly_chart(fig_conf_hist, use_container_width=True)

with col2:
    # Confidence over time
    daily_conf = df.groupby("date")["confidence"].mean().reset_index()
    fig_conf_time = px.line(
        daily_conf,
        x="date",
        y="confidence",
        markers=True,
        title="Average Confidence Over Time",
    )
    fig_conf_time.update_layout(height=300, margin=dict(t=40, b=40, l=40, r=40))
    st.plotly_chart(fig_conf_time, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Recent Queries Table
# ─────────────────────────────────────────────────────────────

st.subheader("📝 Recent Queries")

recent_df = df.sort_values("timestamp", ascending=False).head(20)
display_df = recent_df[["timestamp", "query", "strategy", "confidence", "response_time_ms"]].copy()
display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.1%}")
display_df["response_time_ms"] = display_df["response_time_ms"].apply(lambda x: f"{x:.0f}ms")
display_df.columns = ["Timestamp", "Query", "Strategy", "Confidence", "Response Time"]

st.dataframe(display_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("📥 Export Data")

col1, col2 = st.columns([1, 3])

with col1:
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"analytics_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

with col2:
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
