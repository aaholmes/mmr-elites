#!/usr/bin/env python3
"""
Generate and score LLM responses using Gemini 2.5 Flash.

Produces responses.json for use by llm_response_selection.py.
Requires GEMINI_API_KEY environment variable.

Each response is scored individually in a separate API call to eliminate
order/contrast bias that arises from scoring all responses together.
"""

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

PROMPT = "What strategies help someone learn to code?"

GENERATION_INSTRUCTION = (
    "Give me 25 diverse strategies for learning to code. Each should be 2-3 sentences. "
    "Cover a wide range of approaches: self-study, social learning, project-based, "
    "formal education, gamification, fundamentals-first, mentorship, teaching others, "
    "open-source contribution, reading code, pair programming, etc. "
    "Make each response specific and actionable, not generic platitudes. "
    "Vary the quality naturally -- some should be excellent, some good, some mediocre."
)

SCORING_INSTRUCTION_SINGLE = (
    'Rate the following response to the question: "{PROMPT}"\n\n'
    "Score it on a 1-10 scale considering:\n"
    "- Helpfulness: Does it give useful, practical advice?\n"
    "- Specificity: Does it go beyond generic platitudes?\n"
    "- Actionability: Could someone follow this advice immediately?\n\n"
    "The response:\n"
)


class Response(BaseModel):
    text: str


class ResponseList(BaseModel):
    responses: list[Response]


class SingleScore(BaseModel):
    score: float
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


def score_single_response(client, model: str, response_text: str) -> dict:
    """Score a single response, returning {"score": float, "reasoning": str}."""
    prompt = build_single_scoring_prompt(response_text)
    result = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SingleScore,
        ),
    )
    return json.loads(result.text)


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
