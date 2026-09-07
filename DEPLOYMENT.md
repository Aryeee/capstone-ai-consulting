# Production Deployment Guide

## Prerequisites

- Python 3.9+
- Node.js 20+ (for Next.js deployment)
- Git

## Local Development Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd capstone-ai-consulting
```

### 2. Python/Streamlit Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Build the case index
python -c "from src.case_db import build_all; build_all()"

# Run the Streamlit app
streamlit run app.py
```

The app will be available at `http://localhost:8501`

### 3. Next.js/Vercel Setup

```bash
cd vercel-app

# Install dependencies
npm install

# Pre-compute embeddings (if not done)
cd ..
python scripts/precompute_embeddings.py

# Return to vercel-app
cd vercel-app

# Run development server
npm run dev
```

The app will be available at `http://localhost:3000`

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_predictor.py

# Run with verbose output
pytest -v
```

## Code Quality

```bash
# Format code
black src/

# Lint
flake8 src/

# Type check
mypy src/ --ignore-missing-imports
```

## Deployment

### Vercel Deployment (Recommended for Next.js)

1. **Install Vercel CLI:**
   ```bash
   npm install -g vercel
   ```

2. **Deploy:**
   ```bash
   cd vercel-app
   vercel
   ```

3. **Set up CI/CD:**
   - Push to GitHub
   - Connect repository to Vercel
   - Set up GitHub secrets:
     - `VERCEL_TOKEN`
     - `VERCEL_ORG_ID`
     - `VERCEL_PROJECT_ID`

### Streamlit Cloud Deployment

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Select `app.py` as the main file
5. Deploy

### Docker Deployment (Optional)

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "from src.case_db import build_all; build_all()"

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t strategy-assistant .
docker run -p 8501:8501 strategy-assistant
```

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env` with your settings.

## Performance Optimization

### 1. Model Caching
Models are cached automatically. To specify cache directory:
```bash
export HF_HOME=/path/to/cache
```

### 2. FAISS Index
The FAISS index is built once and persisted to `data/faiss.index`. Rebuild only when cases change.

### 3. Streamlit Caching
The app uses `@st.cache_resource` to cache the index and model loading.

## Monitoring

### Analytics Dashboard

Run the analytics dashboard:
```bash
streamlit run src/analytics_dashboard.py --server.port=8502
```

### Logs

Logs are written to stdout by default. Configure file logging in production:

```python
from src.logging_config import setup_logging

logger = setup_logging(
    level="INFO",
    log_file=Path("logs/app.log")
)
```

## Security Considerations

1. **Input Validation:** All user inputs are validated and sanitized
2. **Rate Limiting:** Implement rate limiting for production (use Redis)
3. **HTTPS:** Always use HTTPS in production
4. **Secrets:** Never commit secrets to Git
5. **Dependencies:** Regularly update dependencies for security patches

## Troubleshooting

### Model download issues
If model download fails, manually download:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```

### FAISS installation issues on Windows
Use pre-built wheels:
```bash
pip install faiss-cpu --no-cache-dir
```

### Memory issues
Reduce batch size in embedding:
```python
embeddings = embed_cases(cases, batch_size=16)
```

## Backup and Maintenance

### Backup critical files
- `data/cases.json` - case database
- `data/faiss.index` - FAISS index
- `data/cases.pkl` - case metadata

### Update cases
1. Edit `data/cases.json`
2. Rebuild index: `python -c "from src.case_db import build_all; build_all()"`
3. For Vercel: `python scripts/precompute_embeddings.py`

## Support

For issues or questions:
- Check GitHub Issues
- Review documentation in README.md
- Contact: [support email]
