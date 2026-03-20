#!/usr/bin/env python3
"""
LLM Response Selection with MMR-Elites
=======================================

Demonstrates MMR-Elites applied to selecting diverse, high-quality text
responses -- connecting the algorithm back to its information retrieval roots.

Given ~25 candidate responses to a prompt (generated and scored by Gemini),
we:
1. Embed them using sentence-transformers (all-MiniLM-L6-v2, 384-dim)
2. Use the Rust MMRSelector to pick diverse top-K
3. Compare with naive top-K by quality alone
4. Show that MMR produces a more diverse selection at minimal quality cost

This is single-shot selection -- no evolution loop. Just the MMR selector
applied to text embeddings, exactly as Carbonell & Goldstein (1998) intended.

Responses are loaded from responses.json (pre-generated, no API key needed).
To regenerate: GEMINI_API_KEY=... python examples/generate_responses.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("sentence-transformers required. Install with:")
    print('  pip install -e ".[examples]"')
    sys.exit(1)

try:
    import mmr_elites_rs
except ImportError:
    print("Rust backend required. Build with:")
    print("  maturin develop --release")
    sys.exit(1)

from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_responses(path: str) -> tuple[str, list[dict]]:
    """Load prompt and responses from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data["prompt"], data["responses"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def embed_responses(texts: list[str]) -> np.ndarray:
    """Embed texts using sentence-transformers, returning L2-normalized vectors."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float64)


def select_top_k(quality: np.ndarray, k: int) -> np.ndarray:
    """Select top-K indices by quality (descending)."""
    return np.argsort(quality)[-k:][::-1]


def compute_diversity(embeddings: np.ndarray, indices: np.ndarray) -> float:
    """Mean pairwise cosine distance among selected embeddings."""
    subset = embeddings[indices]
    # For L2-normalized vectors: cosine_distance = 1 - dot(a, b)
    dists = cdist(subset, subset, metric="cosine")
    n = len(indices)
    # Mean of upper triangle (excluding diagonal)
    return dists[np.triu_indices(n, k=1)].mean()


def print_results(
    title: str,
    indices: np.ndarray,
    responses: list[dict],
    quality: np.ndarray,
    diversity: float,
) -> None:
    """Print a labeled selection with quality info."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  Top-1 quality:    {quality[indices[0]]:.3f}")
    print(f"  Mean quality:     {quality[indices].mean():.3f}")
    print(f"  Cosine diversity: {diversity:.3f}")
    print(f"{'=' * 70}")

    for rank, idx in enumerate(indices, 1):
        r = responses[idx]
        text_preview = r["text"][:80] + ("..." if len(r["text"]) > 80 else "")
        print(f"  {rank}. (q={r['quality']:.2f}) {text_preview}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    k: int = 10, lambda_val: float = 0.5, responses_path: str | None = None
) -> None:
    if responses_path is None:
        responses_path = os.path.join(os.path.dirname(__file__), "responses.json")

    if not os.path.exists(responses_path):
        print(f"Error: {responses_path} not found.")
        print(
            "Generate it with: GEMINI_API_KEY=... python examples/generate_responses.py"
        )
        sys.exit(1)

    prompt, responses = load_responses(responses_path)
    texts = [r["text"] for r in responses]
    quality = np.array([r["quality"] for r in responses], dtype=np.float64)

    print(f'Prompt: "{prompt}"')
    print(f"Candidates: {len(responses)} responses")
    print(f"Selecting: top {k} responses (lambda={lambda_val})")
    print("\nEmbedding responses with all-MiniLM-L6-v2 (384-dim)...")

    embeddings = embed_responses(texts)

    # --- Naive top-K by quality ---
    topk_indices = select_top_k(quality, k)
    topk_diversity = compute_diversity(embeddings, topk_indices)

    # --- MMR-Elites selection ---
    # Sentence-transformer embeddings are L2-normalized, so the Rust selector's
    # Euclidean distance is monotonically related to cosine distance:
    #   ||a - b||^2 = 2(1 - cos(a, b))
    # This means MMR's diversity term correctly maximizes cosine spread.
    selector = mmr_elites_rs.MMRSelector(k, lambda_val)
    mmr_indices = selector.select(quality, embeddings)
    mmr_diversity = compute_diversity(embeddings, mmr_indices)

    # --- Results ---
    print_results(
        "Naive Top-K (by quality)", topk_indices, responses, quality, topk_diversity
    )
    print_results(
        "MMR-Elites Selection", mmr_indices, responses, quality, mmr_diversity
    )

    # --- Summary comparison ---
    topk_q = quality[topk_indices]
    mmr_q = quality[mmr_indices]
    quality_ratio = mmr_q.mean() / topk_q.mean()

    print("=" * 70)
    print("  Summary Comparison")
    print("=" * 70)
    print(f"  {'Metric':<22s} {'Top-K':>10s} {'MMR-Elites':>12s}")
    print(f"  {'-' * 22} {'-' * 10} {'-' * 12}")
    print(f"  {'Top-1 quality':<22s} {topk_q[0]:>10.3f} {mmr_q[0]:>12.3f}")
    print(f"  {'Mean quality':<22s} {topk_q.mean():>10.3f} {mmr_q.mean():>12.3f}")
    print(f"  {'Cosine diversity':<22s} {topk_diversity:>10.3f} {mmr_diversity:>12.3f}")
    print("=" * 70)
    diversity_pct = (mmr_diversity / topk_diversity - 1) * 100
    print(
        f"\n  MMR-Elites achieves {diversity_pct:.0f}% higher diversity while "
        f"maintaining {quality_ratio:.0%} of naive top-K quality.\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM Response Selection with MMR-Elites"
    )
    parser.add_argument(
        "--k", type=int, default=10, help="Number of responses to select"
    )
    parser.add_argument(
        "--lambda", dest="lambda_val", type=float, default=0.5, help="MMR lambda"
    )
    parser.add_argument(
        "--responses",
        dest="responses_path",
        type=str,
        default=None,
        help="Path to responses JSON file",
    )
    args = parser.parse_args()
    main(k=args.k, lambda_val=args.lambda_val, responses_path=args.responses_path)
