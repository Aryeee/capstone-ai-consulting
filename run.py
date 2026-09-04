#!/usr/bin/env python
"""Boot script: installs deps, builds index, launches UI."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    print("=" * 60)
    print("  AI-Powered Strategy Consulting Assistant — Setup")
    print("=" * 60)

    # 1. Build the FAISS index from cases.json
    print("\n[1/2] Building embedding index …")
    sys.path.insert(0, str(ROOT))
    from src.case_db import build_all
    build_all()

    # 2. Launch Streamlit
    print("\n[2/2] Launching Streamlit UI …")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
        "--server.headless", "true",
    ])


if __name__ == "__main__":
    main()
