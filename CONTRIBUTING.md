# Contributing to MMR-Elites

Thank you for your interest in contributing!

## Development Setup

```bash
# Clone and setup
git clone https://github.com/aaholmes/mmr-elites.git
cd mmr-elites

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install development dependencies
pip install -e ".[dev]"

# Build Rust backend
maturin develop --release

# Run tests
pytest tests/ -v
```

## Code Style

We use:
- **black** for Python formatting
- **isort** for import sorting
- **rustfmt** for Rust formatting

```bash
# Format code
black mmr_elites/ tests/ experiments/
isort mmr_elites/ tests/ experiments/

# Check formatting
black --check mmr_elites/
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# With coverage
pytest tests/ --cov=mmr_elites --cov-report=html
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and formatting
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Questions?

Open an issue or reach out to the maintainers.
