#!/usr/bin/env python3
"""
LLM Response Selection with MMR-Elites
=======================================

Demonstrates MMR-Elites applied to selecting diverse, high-quality text
responses -- connecting the algorithm back to its information retrieval roots.

Given ~25 candidate responses to a prompt, we:
1. Embed them using sentence-transformers (all-MiniLM-L6-v2, 384-dim)
2. Use the Rust MMRSelector to pick diverse top-K
3. Compare with naive top-K by quality alone
4. Show that MMR covers all strategy clusters while top-K is redundant

This is single-shot selection -- no evolution loop. Just the MMR selector
applied to text embeddings, exactly as Carbonell & Goldstein (1998) intended.
"""

import argparse
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
# Prompt and candidate responses
# ---------------------------------------------------------------------------

PROMPT = "What strategies help someone learn to code?"

# ~25 responses across 7 semantic clusters, with varying quality.
# Quality scores are designed so naive top-8 draws 4+ from "projects" cluster.
RESPONSES = [
    # --- Cluster: projects (high quality, will dominate naive top-K) ---
    {
        "text": "Build real projects from day one. Start with a personal website, "
        "then a CLI tool, then something with a database. Each project forces you "
        "to solve problems no tutorial covers.",
        "quality": 0.95,
        "cluster": "projects",
    },
    {
        "text": "Pick a project you actually care about -- a budget tracker, a game, "
        "a bot for your group chat. Intrinsic motivation beats any curriculum.",
        "quality": 0.93,
        "cluster": "projects",
    },
    {
        "text": "Clone apps you use daily. Rebuild a simplified Twitter or Spotify. "
        "Reverse-engineering familiar products builds deep intuition for architecture.",
        "quality": 0.91,
        "cluster": "projects",
    },
    {
        "text": "Contribute to open-source side projects on GitHub. Even small PRs "
        "teach you version control, code review, and reading others' code.",
        "quality": 0.89,
        "cluster": "projects",
    },
    # --- Cluster: courses (medium-high quality) ---
    {
        "text": "Follow a structured course like CS50 or freeCodeCamp. A good "
        "curriculum removes decision fatigue about what to learn next.",
        "quality": 0.88,
        "cluster": "courses",
    },
    {
        "text": "Take an interactive course (Codecademy, Exercism) that gives "
        "immediate feedback. Fast feedback loops accelerate learning.",
        "quality": 0.86,
        "cluster": "courses",
    },
    {
        "text": "Watch video tutorials on YouTube or Udemy, but always pause and "
        "code along. Passive watching creates an illusion of understanding.",
        "quality": 0.80,
        "cluster": "courses",
    },
    # --- Cluster: open-source (medium quality) ---
    {
        "text": "Read open-source code on GitHub. Study how popular libraries are "
        "structured -- you'll absorb patterns by osmosis.",
        "quality": 0.85,
        "cluster": "open-source",
    },
    {
        "text": "Find a beginner-friendly open-source project (look for 'good first "
        "issue' labels) and submit a pull request. Real-world codebases teach you "
        "things tutorials never will.",
        "quality": 0.84,
        "cluster": "open-source",
    },
    {
        "text": "Review pull requests on projects you follow. Reading code review "
        "discussions is like eavesdropping on senior engineers teaching.",
        "quality": 0.78,
        "cluster": "open-source",
    },
    # --- Cluster: mentorship (medium quality) ---
    {
        "text": "Find a mentor or study group. Having someone to ask questions "
        "when you're stuck can save hours of frustration.",
        "quality": 0.83,
        "cluster": "mentorship",
    },
    {
        "text": "Pair-program with someone more experienced. Watching how they "
        "debug, navigate docs, and break down problems teaches tacit skills.",
        "quality": 0.82,
        "cluster": "mentorship",
    },
    {
        "text": "Join a coding bootcamp or local meetup. Structured social "
        "accountability prevents the slow fade that kills self-study.",
        "quality": 0.76,
        "cluster": "mentorship",
    },
    {
        "text": "Ask questions on Stack Overflow and Discord servers. Learning "
        "to formulate good questions is itself a core engineering skill.",
        "quality": 0.72,
        "cluster": "mentorship",
    },
    # --- Cluster: teaching (medium quality) ---
    {
        "text": "Teach what you learn -- write blog posts, record videos, or "
        "explain concepts to friends. Teaching forces deep understanding.",
        "quality": 0.81,
        "cluster": "teaching",
    },
    {
        "text": "Rubber-duck debug: explain your code line-by-line to an imaginary "
        "audience. Verbalization exposes gaps in your mental model.",
        "quality": 0.77,
        "cluster": "teaching",
    },
    {
        "text": "Mentor a complete beginner once you're a few months in. You'll "
        "discover which concepts you only half-understand.",
        "quality": 0.73,
        "cluster": "teaching",
    },
    # --- Cluster: games/challenges (medium-low quality) ---
    {
        "text": "Solve coding challenges on LeetCode or HackerRank to sharpen "
        "problem-solving skills. Start easy and progress gradually.",
        "quality": 0.79,
        "cluster": "games",
    },
    {
        "text": "Try game-based platforms like CodeCombat or Screeps where you "
        "write code to control characters. Gamification sustains motivation.",
        "quality": 0.75,
        "cluster": "games",
    },
    {
        "text": "Participate in hackathons. The time pressure forces you to ship "
        "something imperfect, which cures perfectionism paralysis.",
        "quality": 0.74,
        "cluster": "games",
    },
    {
        "text": "Do Advent of Code each December. Daily bite-sized puzzles build "
        "consistency and expose you to algorithmic thinking.",
        "quality": 0.70,
        "cluster": "games",
    },
    # --- Cluster: CS fundamentals (lower quality, but distinct strategy) ---
    {
        "text": "Learn data structures and algorithms early. Understanding how "
        "hash maps and trees work makes you faster at everything else.",
        "quality": 0.71,
        "cluster": "fundamentals",
    },
    {
        "text": "Study how computers actually work -- memory, CPU, networking. "
        "A mental model of the machine makes debugging less mysterious.",
        "quality": 0.68,
        "cluster": "fundamentals",
    },
    {
        "text": "Read 'Structure and Interpretation of Computer Programs' or "
        "'The Little Schemer.' Foundational CS texts rewire how you think "
        "about computation.",
        "quality": 0.65,
        "cluster": "fundamentals",
    },
    {
        "text": "Learn version control (Git) immediately. It's not glamorous, "
        "but every professional workflow depends on it.",
        "quality": 0.63,
        "cluster": "fundamentals",
    },
]


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
    """Print a labeled selection with quality and cluster info."""
    clusters = [responses[i]["cluster"] for i in indices]
    unique_clusters = sorted(set(clusters))

    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  Mean quality:    {quality[indices].mean():.3f}")
    print(f"  Cosine diversity: {diversity:.3f}")
    print(f"  Unique clusters:  {len(unique_clusters)}/7 {unique_clusters}")
    print(f"{'=' * 70}")

    for rank, idx in enumerate(indices, 1):
        r = responses[idx]
        text_preview = r["text"][:80] + ("..." if len(r["text"]) > 80 else "")
        print(f"  {rank}. [{r['cluster']:<12s}] (q={r['quality']:.2f}) {text_preview}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(k: int = 8, lambda_val: float = 0.5) -> None:
    texts = [r["text"] for r in RESPONSES]
    quality = np.array([r["quality"] for r in RESPONSES], dtype=np.float64)

    print(f'Prompt: "{PROMPT}"')
    print(f"Candidates: {len(RESPONSES)} responses across 7 strategy clusters")
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
        "Naive Top-K (by quality)", topk_indices, RESPONSES, quality, topk_diversity
    )
    print_results(
        "MMR-Elites Selection", mmr_indices, RESPONSES, quality, mmr_diversity
    )

    # --- Summary comparison ---
    topk_clusters = len(set(RESPONSES[i]["cluster"] for i in topk_indices))
    mmr_clusters = len(set(RESPONSES[i]["cluster"] for i in mmr_indices))
    quality_ratio = quality[mmr_indices].mean() / quality[topk_indices].mean()

    print("=" * 70)
    print("  Summary Comparison")
    print("=" * 70)
    print(f"  {'Metric':<22s} {'Top-K':>10s} {'MMR-Elites':>12s}")
    print(f"  {'-' * 22} {'-' * 10} {'-' * 12}")
    print(
        f"  {'Mean quality':<22s} {quality[topk_indices].mean():>10.3f}"
        f" {quality[mmr_indices].mean():>12.3f}"
    )
    print(f"  {'Cosine diversity':<22s} {topk_diversity:>10.3f} {mmr_diversity:>12.3f}")
    print(f"  {'Unique clusters':<22s} {topk_clusters:>10d} {mmr_clusters:>12d}")
    print("=" * 70)
    print(
        f"\n  MMR-Elites covers {mmr_clusters}/7 strategy clusters while "
        f"maintaining {quality_ratio:.0%} of naive top-K quality.\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM Response Selection with MMR-Elites"
    )
    parser.add_argument(
        "--k", type=int, default=8, help="Number of responses to select"
    )
    parser.add_argument(
        "--lambda", dest="lambda_val", type=float, default=0.5, help="MMR lambda"
    )
    args = parser.parse_args()
    main(k=args.k, lambda_val=args.lambda_val)
