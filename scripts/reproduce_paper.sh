#!/bin/bash
# reproduce_paper.sh — Reproduce all experiments and figures for the MMR-Elites paper
#
# Usage:
#   ./scripts/reproduce_paper.sh          # Full experiments (10 seeds, ~4-8 hours)
#   ./scripts/reproduce_paper.sh --quick  # Quick test (2 seeds, ~10 minutes)
#
# Prerequisites:
#   1. Install Rust toolchain: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
#   2. Build the Rust backend: maturin develop --release
#   3. Install Python deps: pip install -e ".[dev]"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MODE="${1:---default}"

echo "=============================================="
echo "MMR-Elites Paper Reproduction Pipeline"
echo "=============================================="
echo "Project dir: $PROJECT_DIR"
echo "Mode: $MODE"
echo ""

# Step 0: Verify environment
echo "[0/5] Verifying environment..."
python -c "import mmr_elites_rs; print('Rust backend: OK')" || {
    echo "ERROR: Rust backend not built. Run: maturin develop --release"
    exit 1
}
python -c "import mmr_elites; print('Python package: OK')" || {
    echo "ERROR: Python package not installed. Run: pip install -e '.[dev]'"
    exit 1
}
echo ""

# Step 1: Run tests
echo "[1/5] Running tests..."
PYTHONPATH=. pytest tests/ -v --tb=short || {
    echo "WARNING: Some tests failed. Continuing anyway..."
}
echo ""

# Step 2: Run experiments
echo "[2/5] Running experiments..."
RESULTS_DIR="results/paper_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

if [ "$MODE" = "--quick" ]; then
    echo "  Quick mode: 2 seeds, 200 generations"
    PYTHONPATH=. python experiments/run_all.py --quick --output-dir "$RESULTS_DIR"
elif [ "$MODE" = "--full" ]; then
    echo "  Full mode: 10 seeds, 2000 generations"
    PYTHONPATH=. python experiments/run_all.py --full --output-dir "$RESULTS_DIR"
else
    echo "  Default mode: 5 seeds, 1000 generations"
    PYTHONPATH=. python experiments/run_all.py --output-dir "$RESULTS_DIR"
fi
echo ""

# Step 3: Generate figures
echo "[3/5] Generating figures..."
# Find the most recent run directory inside RESULTS_DIR
LATEST_RUN=$(ls -td "$RESULTS_DIR"/run_* 2>/dev/null | head -1)
if [ -z "$LATEST_RUN" ]; then
    LATEST_RUN="$RESULTS_DIR"
fi
PYTHONPATH=. python paper/plot_all.py --results-dir "$LATEST_RUN"
echo ""

# Step 4: Compile paper
echo "[4/5] Compiling paper..."
cd paper
if command -v pdflatex &> /dev/null; then
    make all 2>&1 || echo "WARNING: LaTeX compilation failed. Install texlive-full for PDF output."
else
    echo "  SKIP: pdflatex not found. Install texlive for PDF compilation."
fi
cd "$PROJECT_DIR"
echo ""

# Step 5: Summary
echo "[5/5] Summary"
echo "=============================================="
echo "Results:  $RESULTS_DIR"
echo "Figures:  paper/figures/"
if [ -f paper/main.pdf ]; then
    echo "Paper:    paper/main.pdf"
fi
echo ""
echo "Done!"
