#!/usr/bin/env python3
"""
Generate and score LLM responses using Gemini 2.5 Flash.

Produces responses.json for use by llm_response_selection.py.
Requires GEMINI_API_KEY environment variable.

Each response is scored individually in a separate API call to eliminate
order/contrast bias that arises from scoring all responses together.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

try:
    from google import genai
except ImportError:
    print("google-genai required. Install with:")
    print('  pip install "google-genai>=1.0"')
    sys.exit(1)

from pydantic import BaseModel

PROMPT = "How should I approach fundraising for my startup?"

GENERATION_INSTRUCTION = (
    "Give me 50 pieces of advice about fundraising for a startup. "
    "Each should be 2-3 sentences. "
    "Don't worry about covering different topics -- it's fine if multiple responses "
    "address similar themes like pitch decks, investor meetings, or valuations. "
    "Vary the quality naturally -- some should be genuinely novel insights that only "
    "experienced founders would know, some should be decent but conventional wisdom, "
    "and some should be vague platitudes or shallow advice."
)

SCORING_INSTRUCTION_SINGLE = (
    'Rate the following response to the question: "{PROMPT}"\n\n'
    "Score each criterion independently on a 1-10 scale:\n\n"
    "- Novelty (weight: 0.3): Does this offer a non-obvious insight, or could anyone "
    "have said this? Truly original earns 8+. Generic listicle advice earns 1-4.\n"
    "- Precision (weight: 0.3): Does it name specific tools, techniques, numbers, or "
    'steps? Vague advice ("network more") scores low. Concrete advice with named '
    "methods or metrics scores high.\n"
    "- Depth (weight: 0.2): Does it explain WHY the strategy works, not just WHAT to "
    "do? Surface-level earns 1-4, mechanistic reasoning earns 7+.\n"
    "- Completeness (weight: 0.2): Could someone act on this immediately without "
    "needing to look anything else up?\n\n"
    "Be a tough but fair grader. Most responses should NOT score above 7 on novelty.\n\n"
    "The response:\n"
)


class Response(BaseModel):
    text: str


class ResponseList(BaseModel):
    responses: list[Response]


class SingleScore(BaseModel):
    novelty: float
    precision: float
    depth: float
    completeness: float
    reasoning: str


def normalize_scores(raw_scores: list[float]) -> list[float]:
    """Normalize raw scores to [0, 1]. All-same → 0.5, empty → []."""
    if not raw_scores:
        return []
    if len(raw_scores) == 1:
        return [0.5]
    min_s, max_s = min(raw_scores), max(raw_scores)
    if max_s > min_s:
        return [(s - min_s) / (max_s - min_s) for s in raw_scores]
    return [0.5] * len(raw_scores)


def build_single_scoring_prompt(response_text: str) -> str:
    """Build a scoring prompt for a single response."""
    return SCORING_INSTRUCTION_SINGLE.format(PROMPT=PROMPT) + response_text


def generate_responses(client, model: str) -> list[str]:
    """Generate candidate responses using the LLM."""
    gen_result = client.models.generate_content(
        model=model,
        contents=GENERATION_INSTRUCTION,
        config=genai.types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ResponseList,
        ),
    )
    response_list = json.loads(gen_result.text)
    return [r["text"] for r in response_list["responses"]]


SCORE_WEIGHTS = {"novelty": 0.3, "precision": 0.3, "depth": 0.2, "completeness": 0.2}


def compute_weighted_score(scores: dict) -> float:
    """Compute weighted total from per-criterion scores."""
    return sum(scores[k] * w for k, w in SCORE_WEIGHTS.items())


def score_single_response(client, model: str, response_text: str) -> dict:
    """Score a single response with weighted criteria.

    Returns dict with per-criterion scores, weighted total, and reasoning.
    """
    prompt = build_single_scoring_prompt(response_text)
    result = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SingleScore,
        ),
    )
    parsed = json.loads(result.text)
    parsed["score"] = compute_weighted_score(parsed)
    return parsed


def score_all_responses(client, model: str, texts: list[str]) -> list[dict]:
    """Score each response individually, printing progress."""
    scores = []
    for i, text in enumerate(texts, 1):
        print(f"  Scoring response {i}/{len(texts)}...")
        scores.append(score_single_response(client, model, text))
    return scores


def build_output(
    prompt: str,
    model: str,
    texts: list[str],
    normalized: list[float],
    scores: list[dict],
) -> dict:
    """Build the output dictionary for JSON serialization."""
    return {
        "prompt": prompt,
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "responses": [
            {
                "text": t,
                "quality": round(q, 3),
                "raw_score": round(s["score"], 2),
                "novelty": s["novelty"],
                "precision": s["precision"],
                "depth": s["depth"],
                "completeness": s["completeness"],
                "score_reasoning": s["reasoning"],
            }
            for t, q, s in zip(texts, normalized, scores)
        ],
    }


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable required.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-flash"

    # Step 1: Generate responses
    print(f"Generating responses with {model}...")
    texts = generate_responses(client, model)
    print(f"  Generated {len(texts)} responses.")

    # Step 2: Score each response individually
    print("Scoring responses...")
    scores = score_all_responses(client, model, texts)

    # Step 3: Normalize and save
    raw_scores = [s["score"] for s in scores]
    normalized = normalize_scores(raw_scores)

    output = build_output(PROMPT, model, texts, normalized, scores)

    out_path = os.path.join(os.path.dirname(__file__), "responses.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(texts)} responses to {out_path}")
    print(f"Quality range: {min(normalized):.3f} - {max(normalized):.3f}")


if __name__ == "__main__":
    main()
