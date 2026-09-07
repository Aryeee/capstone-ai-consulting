# AI-Powered Strategy Consulting Assistant

> **Enterprise-Grade Case-Based Reasoning System** — Evidence-based strategy recommendations powered by semantic search and explainable AI

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

An AI-powered consulting assistant that leverages **case-based reasoning** to provide evidence-based strategy recommendations. The system uses neural embeddings to find similar historical business cases and provides detailed explanations of its reasoning process.

### Key Features

✨ **Semantic Search** — Neural embeddings (sentence-transformers) find similar cases  
📊 **Evidence-Based** — Recommendations backed by historical outcomes  
🔍 **Explainable AI** — Detailed reasoning for every recommendation  
💰 **ROI Prediction** — Estimated returns with confidence intervals  
⚡ **Dual Deployment** — Python/Streamlit + Next.js/Vercel options  
🧪 **Production-Ready** — Tests, logging, security, analytics, CI/CD  

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   User Query (Business Problem)               │
└──────────────────────┬───────────────────────────────────────┘
                       │
         ┌─────────────▼──────────────┐
         │  Embedding & Retrieval     │
         │  • Sentence Transformer    │
         │  • FAISS Vector Search     │
         │  • Top-K Similar Cases     │
         └─────────────┬──────────────┘
                       │
         ┌─────────────▼──────────────┐
         │  Prediction                │
         │  • Weighted k-NN Vote      │
         │  • Strategy Classification │
         │  • ROI Estimation          │
         └─────────────┬──────────────┘
                       │
         ┌─────────────▼──────────────┐
         │  Explainability            │
         │  • Similarity Breakdown    │
         │  • Feature Matching        │
         │  • Confidence Assessment   │
         └─────────────┬──────────────┘
                       │
         ┌─────────────▼──────────────┐
         │  Structured Recommendation │
         │  + Detailed Explanation    │
         └────────────────────────────┘
```

## 📊 What's Inside

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Python Backend** | Streamlit, FAISS, sentence-transformers | Full-featured web app |
| **Next.js Frontend** | React 19, TypeScript, Transformers.js | Browser-based ML deployment |
| **Case Database** | JSON, 28 structured cases | Historical business scenarios |
| **Embeddings** | all-MiniLM-L6-v2 (384-dim) | Semantic similarity search |
| **Analytics** | Plotly, Streamlit dashboard | Usage tracking & insights |
| **Tests** | pytest, 90%+ coverage | Comprehensive test suite |
| **CI/CD** | GitHub Actions | Automated testing & deployment |

## 🚀 Quick Start

### Option 1: Python/Streamlit (Recommended for Development)

```bash
# Clone repository
git clone <repository-url>
cd capstone-ai-consulting

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Build case index (first time only)
python -c "from src.case_db import build_all; build_all()"

# Launch app
streamlit run app.py
```

Visit `http://localhost:8501`

### Option 2: Next.js/Vercel (Production Deployment)

```bash
cd vercel-app

# Install dependencies
npm install

# Pre-compute embeddings (first time only)
cd .. && python scripts/precompute_embeddings.py && cd vercel-app

# Run development server
npm run dev
```

Visit `http://localhost:3000`

## 🧪 Running Tests

```bash
# Run all tests with coverage
pytest --cov=src --cov-report=html

# Run specific test suite
pytest tests/test_predictor.py -v

# Run integration tests
pytest tests/test_integration.py

# Run security tests
pytest tests/test_security.py
```

View coverage report: `open htmlcov/index.html`

## 📈 Analytics Dashboard

Track usage, popular queries, and performance:

```bash
streamlit run src/analytics_dashboard.py --server.port=8502
```

Features:
- Query volume over time
- Strategy distribution
- Response time metrics
- Confidence analysis
- Industry breakdown

## 🛠️ Development

### Code Quality

```bash
# Format code
black src/

# Lint
flake8 src/

# Type check
mypy src/ --ignore-missing-imports
```

### Project Structure

```
capstone-ai-consulting/
├── src/                          # Core Python modules
│   ├── case_db.py               # FAISS indexing & retrieval
│   ├── predictor.py             # Weighted k-NN prediction
│   ├── explainer.py             # Explainability layer
│   ├── schema.py                # Data models
│   ├── security.py              # Input validation
│   ├── error_handling.py        # Error management
│   ├── logging_config.py        # Logging setup
│   └── analytics_dashboard.py   # Analytics UI
├── tests/                        # Comprehensive test suite
│   ├── test_case_db.py
│   ├── test_predictor.py
│   ├── test_explainer.py
│   ├── test_integration.py
│   └── test_security.py
├── data/
│   ├── cases.json               # 28 business cases
│   ├── faiss.index              # FAISS vector index
│   └── cases.pkl                # Serialized case metadata
├── vercel-app/                   # Next.js deployment
│   ├── src/
│   │   ├── app/                 # Next.js pages
│   │   └── lib/                 # Client-side ML engine
│   └── public/data/             # Pre-computed embeddings
├── scripts/
│   └── precompute_embeddings.py # Generate embeddings for Next.js
├── app.py                        # Main Streamlit app
├── app_enhanced.py              # Production app with logging
├── .github/workflows/
│   └── ci-cd.yml                # CI/CD pipeline
├── pyproject.toml               # Python project config
├── DEPLOYMENT.md                # Deployment guide
├── CONTRIBUTING.md              # Contribution guidelines
└── CHANGELOG.md                 # Version history
```

## 🔒 Security

- ✅ Input validation and sanitization
- ✅ XSS and injection attack prevention
- ✅ Rate limiting support (production)
- ✅ Secure error handling (no internal details exposed)
- ✅ Environment variable management

## 📦 Deployment

### Vercel (Automated)

1. Push to GitHub
2. Connect repository to Vercel
3. Set environment variables (if needed)
4. Deploy automatically on push to `main`

### Docker

```dockerfile
docker build -t strategy-assistant .
docker run -p 8501:8501 strategy-assistant
```

### Streamlit Cloud

1. Push to GitHub
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Select `app.py` as entry point
4. Deploy

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

## 📚 How It Works

### 1. Case Representation
Each case is encoded into a 384-dimensional vector using `all-MiniLM-L6-v2`, capturing semantic meaning of the business problem.

### 2. Semantic Retrieval
User queries are embedded with the same model. FAISS performs efficient cosine similarity search to find the top-k most similar historical cases.

### 3. Strategy Prediction
A similarity-weighted k-NN vote aggregates strategy labels from retrieved cases. ROI estimates use weighted averages ± standard deviation.

### 4. Explainability
For each retrieved case:
- **Similarity score** (cosine distance)
- **Key overlapping phrases** between query and case
- **Shared vs differing features** (industry, size, region)
- **Strategy drivers** — which cases voted for which strategy
- **Outcome evidence** — historical outcomes with confidence

## 🎓 Data Schema

Each case contains:

| Field | Type | Description |
|-------|------|-------------|
| `case_id` | string | Unique identifier |
| `title` | string | Human-readable name |
| `industry` | string | Industry category |
| `company_size` | enum | Small / Medium / Large |
| `problem_description` | text | Business problem (main embedding) |
| `key_challenges` | list[str] | Concise challenge bullets |
| `actions_taken` | text | What was implemented |
| `strategy_type` | string | Strategy category |
| `outcome` | enum | Positive / Mixed / Negative |
| `outcome_metrics` | object | ROI, revenue change, timeline |
| `tags` | list[str] | Topic tags |

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes and add tests
4. Run tests and linters
5. Commit (`git commit -m 'feat: add amazing feature'`)
6. Push and create a Pull Request

## 📊 Performance

- **Python Backend**: ~500-1500ms average response time
- **Browser ML**: ~2000-4000ms (first query with model load)
- **Scalability**: Supports 1000+ cases with FAISS
- **Model Size**: 90MB (sentence-transformer)

## 🔮 Roadmap

- [ ] Multi-language support
- [ ] Real-time collaborative filtering
- [ ] Advanced analytics with ML insights
- [ ] API endpoints for programmatic access
- [ ] Integration with external data sources (Airtable, Notion)
- [ ] Enhanced recommendation algorithms (ensemble methods)
- [ ] Mobile app

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **sentence-transformers** — Neural embedding models
- **FAISS** — Efficient similarity search
- **Streamlit** — Rapid web app development
- **Next.js & Vercel** — Modern web deployment
- **Transformers.js** — Browser-based ML

## 📞 Support

- 📧 [Open an issue](https://github.com/your-repo/issues)
- 📖 [Read the docs](./DEPLOYMENT.md)
- 💬 [Discussions](https://github.com/your-repo/discussions)

---

**Built with ❤️ using AI-powered development tools**
