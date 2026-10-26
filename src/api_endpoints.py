"""
FastAPI endpoints for programmatic consulting access.

Provides REST API for queries, batch processing, analytics, and feedback.
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import csv
import io
import uuid
from pathlib import Path

from src.case_db import load_index, retrieve, get_model
from src.hybrid_retrieval import RetrievalFilters, retrieve_hybrid
from src.predictor import predict
from src.explainer import explain
from src.ensemble_predictor import predict_ensemble, load_ensemble
from src.analytics import log_query, store_feedback, get_analytics_stats
from src.query_cache import get_cache


# ─────────────────────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────────────────────


class ConsultRequest(BaseModel):
    """Request for a consultation."""

    query: str = Field(..., min_length=10, description="Business problem description")
    industry: Optional[str] = None
    company_size: Optional[str] = None
    region: Optional[str] = None
    strategy_type: Optional[str] = None
    top_k: int = Field(5, ge=1, le=20)
    use_ensemble: bool = True


class ConsultResponse(BaseModel):
    """Consultation response."""

    query_id: str
    recommended_strategy: str
    strategy_confidence: float
    predicted_outcome: str
    estimated_roi_low: float
    estimated_roi_high: float
    avg_time_to_result_months: float
    supporting_rationale: str
    retrieved_cases: List[dict]
    explanation: Optional[dict] = None


class FeedbackRequest(BaseModel):
    """User feedback on a consultation."""

    query_id: str
    user_rating: int = Field(..., ge=1, le=5)
    helpful: bool
    notes: Optional[str] = None


class BatchConsultItem(BaseModel):
    """Single item in batch consulting request."""

    problem_description: str
    industry: Optional[str] = None
    company_size: Optional[str] = None


class StrategyStatsResponse(BaseModel):
    """Statistics for a strategy."""

    strategy_type: str
    num_cases: int
    positive_count: int
    mixed_count: int
    negative_count: int
    avg_roi: float
    roi_range: tuple[float, float]
    avg_time_to_result: float
    success_rate: float


# ─────────────────────────────────────────────────────────────
# Initialize FastAPI
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Strategy Consulting API",
    description="Case-based reasoning consulting system",
    version="2.0.0",
)


# Load models once at startup
_index = None
_cases = None
_ensemble = None
_cache = None


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    global _index, _cases, _ensemble, _cache
    _index, _cases = load_index()
    _ensemble = load_ensemble()
    _cache = get_cache()


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────


@app.post("/api/consult", response_model=ConsultResponse)
async def consult(request: ConsultRequest) -> ConsultResponse:
    """Get a consultation recommendation."""
    try:
        query_id = str(uuid.uuid4())

        # Build filters
        filters = RetrievalFilters(
            industry=request.industry,
            company_size=request.company_size,
            region=request.region,
            strategy_type=request.strategy_type,
        )

        # Check cache
        cache = get_cache()
        cache_key = {
            "industry": request.industry,
            "company_size": request.company_size,
            "region": request.region,
            "strategy_type": request.strategy_type,
        }
        cached = cache.get(request.query, cache_key)

        if cached:
            embedding, recommendation, retrieved_case_ids = cached
            retrieved = [(c, 0.95) for c in _cases if c.case_id in retrieved_case_ids]
        else:
            # Embed query
            model = get_model()
            query_embedding = model.encode(
                [request.query], normalize_embeddings=True, convert_to_numpy=True
            ).astype("float32")

            # Hybrid retrieval
            retrieved = retrieve_hybrid(
                query_embedding, _index, _cases, filters, request.top_k
            )

            # Predict
            if request.use_ensemble:
                recommendation = predict_ensemble(retrieved, _ensemble)
            else:
                recommendation = predict(retrieved)

            # Cache result
            retrieved_case_ids = [c.case_id for c, _ in retrieved]
            cache.set(
                request.query,
                query_embedding[0],
                recommendation,
                retrieved_case_ids,
                cache_key,
            )

        # Log query
        query_features = {
            "industry": request.industry,
            "company_size": request.company_size,
        }
        log_query(
            query_id,
            request.query,
            retrieved,
            recommendation,
            request.industry,
            request.company_size,
            request.top_k,
        )

        # Explain
        explanation = explain(
            request.query, query_features, retrieved, recommendation
        )

        # Format response
        retrieved_cases = [
            {
                "case_id": c.case_id,
                "title": c.title,
                "industry": c.industry,
                "strategy_type": c.strategy_type,
                "outcome": c.outcome,
                "roi": c.outcome_metrics.roi_estimate,
                "similarity_score": round(score, 3),
            }
            for c, score in retrieved
        ]

        return ConsultResponse(
            query_id=query_id,
            recommended_strategy=recommendation.recommended_strategy,
            strategy_confidence=recommendation.strategy_confidence,
            predicted_outcome=recommendation.predicted_outcome,
            estimated_roi_low=recommendation.estimated_roi_low,
            estimated_roi_high=recommendation.estimated_roi_high,
            avg_time_to_result_months=recommendation.avg_time_to_result_months,
            supporting_rationale=recommendation.supporting_rationale,
            retrieved_cases=retrieved_cases,
            explanation={
                "case_explanations": [
                    {
                        "case_id": ce.case_id,
                        "title": ce.title,
                        "similarity_score": ce.similarity_score,
                        "shared_features": ce.shared_features,
                        "key_overlap_phrases": ce.key_overlap_phrases,
                    }
                    for ce in explanation.case_explanations
                ],
                "strategy_drivers": explanation.strategy_drivers,
                "confidence_note": explanation.confidence_note,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback on a consultation."""
    try:
        store_feedback(request.query_id, request.user_rating, request.helpful, request.notes)
        return {"status": "success", "query_id": request.query_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics")
async def get_analytics() -> dict:
    """Get analytics and usage statistics."""
    try:
        stats = get_analytics_stats()
        cache_stats = get_cache().get_stats()
        return {**stats, "cache": cache_stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/case/{case_id}")
async def get_case(case_id: str) -> dict:
    """Get details for a specific case."""
    try:
        case = next((c for c in _cases if c.case_id == case_id), None)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")

        return {
            "case_id": case.case_id,
            "title": case.title,
            "industry": case.industry,
            "company_size": case.company_size,
            "region": case.region,
            "year": case.year,
            "problem_description": case.problem_description,
            "key_challenges": case.key_challenges,
            "actions_taken": case.actions_taken,
            "strategy_type": case.strategy_type,
            "outcome": case.outcome,
            "outcome_description": case.outcome_description,
            "outcome_metrics": {
                "revenue_change_pct": case.outcome_metrics.revenue_change_pct,
                "time_to_result_months": case.outcome_metrics.time_to_result_months,
                "roi_estimate": case.outcome_metrics.roi_estimate,
            },
            "tags": case.tags,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/strategies")
async def list_strategies() -> dict:
    """List all strategies with statistics."""
    try:
        strategies = {}
        for case in _cases:
            if case.strategy_type not in strategies:
                strategies[case.strategy_type] = {
                    "num_cases": 0,
                    "positive": 0,
                    "mixed": 0,
                    "negative": 0,
                    "rois": [],
                    "times": [],
                }

            strat = strategies[case.strategy_type]
            strat["num_cases"] += 1
            if case.outcome == "Positive":
                strat["positive"] += 1
            elif case.outcome == "Mixed":
                strat["mixed"] += 1
            else:
                strat["negative"] += 1

            strat["rois"].append(case.outcome_metrics.roi_estimate)
            strat["times"].append(case.outcome_metrics.time_to_result_months)

        # Compute aggregates
        result = {}
        for strategy, data in strategies.items():
            import statistics

            result[strategy] = {
                "num_cases": data["num_cases"],
                "outcomes": {
                    "positive": data["positive"],
                    "mixed": data["mixed"],
                    "negative": data["negative"],
                },
                "success_rate": round(data["positive"] / data["num_cases"], 2),
                "avg_roi": round(statistics.mean(data["rois"]), 2),
                "roi_range": (round(min(data["rois"]), 2), round(max(data["rois"]), 2)),
                "avg_time_to_result": round(statistics.mean(data["times"]), 1),
            }

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/batch")
async def batch_consult(
    file: UploadFile = File(...), background_tasks: BackgroundTasks = None
):
    """Process batch of consulting requests from CSV."""
    try:
        contents = await file.read()
        reader = csv.DictReader(io.StringIO(contents.decode("utf-8")))

        results = []
        for row in reader:
            try:
                req = ConsultRequest(
                    query=row.get("problem_description", ""),
                    industry=row.get("industry"),
                    company_size=row.get("company_size"),
                )

                # Process (reuse /api/consult logic)
                response = await consult(req)
                results.append(response.dict())

            except Exception as e:
                results.append({"error": str(e), **row})

        # Export results
        output = io.StringIO()
        if results and "error" not in results[0]:
            writer = csv.DictWriter(
                output,
                fieldnames=[
                    "problem_description",
                    "recommended_strategy",
                    "strategy_confidence",
                    "predicted_outcome",
                    "estimated_roi_low",
                    "estimated_roi_high",
                ],
            )
            writer.writeheader()
            for r in results:
                writer.writerow(
                    {
                        "problem_description": r.get("query", ""),
                        "recommended_strategy": r.get("recommended_strategy", ""),
                        "strategy_confidence": r.get("strategy_confidence", ""),
                        "predicted_outcome": r.get("predicted_outcome", ""),
                        "estimated_roi_low": r.get("estimated_roi_low", ""),
                        "estimated_roi_high": r.get("estimated_roi_high", ""),
                    }
                )

        return {
            "processed": len(results),
            "results": results,
            "csv_export": output.getvalue(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "cases_loaded": len(_cases) if _cases else 0,
        "ensemble_available": _ensemble is not None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
