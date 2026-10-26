"""
Setup script for all improvements: case expansion, model training, caching initialization.

Run this after installing requirements to prepare the system.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.case_db import build_all, load_index
from src.analytics import init_logs
from src.query_cache import get_cache
from src.ensemble_predictor import train_ensemble, save_ensemble, load_ensemble
from scripts.expand_cases import expand_cases


def setup_improvements():
    """Execute full setup for all improvements."""
    print("\n" + "=" * 70)
    print("AI STRATEGY CONSULTING ASSISTANT - IMPROVEMENTS SETUP")
    print("=" * 70)

    # Step 1: Expand case database
    print("\n[1/5] Expanding case database with 35 new cases...")
    try:
        total_cases = expand_cases()
        print(f"✓ Case database expanded to {total_cases} total cases")
    except Exception as e:
        print(f"✗ Case expansion failed: {e}")
        return False

    # Step 2: Rebuild FAISS index with new cases
    print("\n[2/5] Rebuilding FAISS index with expanded case database...")
    try:
        index, cases = build_all()
        print(f"✓ FAISS index rebuilt with {len(cases)} cases")
    except Exception as e:
        print(f"✗ Index rebuild failed: {e}")
        return False

    # Step 3: Train ensemble models
    print("\n[3/5] Training ensemble prediction models...")
    try:
        ensemble = train_ensemble(cases)
        save_ensemble(ensemble)
        print(f"✓ Ensemble models trained and saved")
        print(f"  - Strategy classes: {ensemble.strategy_classes}")
    except Exception as e:
        print(f"✗ Ensemble training failed: {e}")
        return False

    # Step 4: Initialize analytics and feedback logs
    print("\n[4/5] Initializing analytics and feedback tracking...")
    try:
        init_logs()
        print(f"✓ Analytics and feedback logs initialized")
    except Exception as e:
        print(f"✗ Analytics initialization failed: {e}")
        return False

    # Step 5: Initialize query cache
    print("\n[5/5] Initializing query cache...")
    try:
        cache = get_cache()
        stats = cache.get_stats()
        print(f"✓ Query cache initialized")
        print(f"  - Cached queries: {stats['cached_queries']}")
    except Exception as e:
        print(f"✗ Cache initialization failed: {e}")
        return False

    # Summary
    print("\n" + "=" * 70)
    print("SETUP COMPLETE ✓")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Start Streamlit app:")
    print("   streamlit run app.py")
    print("\n2. Start FastAPI server (in separate terminal):")
    print("   python -m src.api_endpoints")
    print("   # API will be available at http://localhost:8000")
    print("   # Docs at http://localhost:8000/docs")
    print("\n3. Run analytics dashboard:")
    print("   streamlit run src/analytics_dashboard.py --server.port=8502")
    print("\n4. Run tests:")
    print("   pytest tests/ -v --cov=src")
    print("\n" + "=" * 70)

    return True


if __name__ == "__main__":
    success = setup_improvements()
    sys.exit(0 if success else 1)
