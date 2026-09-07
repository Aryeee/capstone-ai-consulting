# Contributing to AI Strategy Consulting Assistant

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in Issues
2. If not, create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Error messages or logs

### Suggesting Features

1. Check if the feature has been suggested in Issues
2. Create a new issue with:
   - Clear description of the feature
   - Use case and benefits
   - Possible implementation approach

### Submitting Pull Requests

1. **Fork the repository**

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the code style guidelines
   - Write or update tests
   - Update documentation if needed

4. **Run tests locally**
   ```bash
   pytest
   black src/
   flake8 src/
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation changes
   - `test:` adding or updating tests
   - `refactor:` code refactoring
   - `chore:` maintenance tasks

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Provide a clear description
   - Reference related issues
   - Explain what and why, not just how

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/capstone-ai-consulting.git
cd capstone-ai-consulting

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies including dev tools
pip install -r requirements.txt

# Build index
python -c "from src.case_db import build_all; build_all()"

# Run tests
pytest
```

## Code Style Guidelines

### Python

- Follow [PEP 8](https://pep8.org/)
- Use Black for formatting (line length 100)
- Use type hints where possible
- Write docstrings for all public functions/classes (NumPy style)

Example:
```python
def retrieve(query: str, index: faiss.IndexFlatIP, cases: list[Case],
             top_k: int = 5) -> list[tuple[Case, float]]:
    """
    Embed *query*, search the FAISS index, return top-k (Case, score) pairs.
    
    Parameters
    ----------
    query : str
        User's business problem description
    index : faiss.IndexFlatIP
        Pre-built FAISS index
    cases : list[Case]
        List of all cases
    top_k : int, default=5
        Number of results to return
        
    Returns
    -------
    list[tuple[Case, float]]
        Retrieved cases with similarity scores
    """
    # Implementation
```

### TypeScript/React

- Follow project's ESLint configuration
- Use functional components with hooks
- Use TypeScript for type safety
- Keep components focused and reusable

## Testing Guidelines

### Write Tests For

- New features
- Bug fixes
- Edge cases
- Integration scenarios

### Test Structure

```python
class TestFeature:
    def test_normal_case(self):
        """Test the happy path."""
        # Arrange
        input_data = ...
        
        # Act
        result = function(input_data)
        
        # Assert
        assert result == expected
    
    def test_edge_case(self):
        """Test edge cases."""
        # ...
    
    def test_error_handling(self):
        """Test error conditions."""
        with pytest.raises(ExpectedError):
            function(bad_input)
```

### Test Coverage

- Aim for >80% coverage
- All new code should have tests
- Run coverage report: `pytest --cov=src --cov-report=html`

## Documentation

### Update Documentation When

- Adding new features
- Changing API or behavior
- Fixing bugs (if they reveal unclear docs)

### Documentation Types

- **README.md** - Overview, quick start, features
- **DEPLOYMENT.md** - Production deployment guide
- **Code comments** - Non-obvious logic, complex algorithms
- **Docstrings** - All public functions/classes
- **Examples** - Usage examples in docstrings or separate files

## Adding New Cases

To add cases to the database:

1. Edit `data/cases.json`
2. Follow the schema:
   ```json
   {
     "case_id": "CASE-XXX",
     "title": "Descriptive Title",
     "industry": "Industry Name",
     "company_size": "Small|Medium|Large",
     "region": "Geographic Region",
     "year": 2024,
     "problem_description": "Detailed problem description...",
     "key_challenges": ["Challenge 1", "Challenge 2"],
     "actions_taken": "What was done...",
     "strategy_type": "Strategy Category",
     "outcome": "Positive|Mixed|Negative",
     "outcome_description": "What happened...",
     "outcome_metrics": {
       "revenue_change_pct": 15.0,
       "time_to_result_months": 12,
       "roi_estimate": 2.5
     },
     "tags": ["tag1", "tag2"]
   }
   ```
3. Rebuild indexes:
   ```bash
   python -c "from src.case_db import build_all; build_all()"
   python scripts/precompute_embeddings.py
   ```

## Review Process

1. All PRs require at least one review
2. CI/CD pipeline must pass
3. Reviewer will check:
   - Code quality and style
   - Test coverage
   - Documentation
   - Breaking changes

## Questions?

- Open an issue for questions
- Check existing documentation
- Reach out to maintainers

Thank you for contributing!
