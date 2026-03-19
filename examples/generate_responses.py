#!/usr/bin/env python3
"""
Generate and score LLM responses using Gemini 2.5 Flash.

Produces responses.json for use by llm_response_selection.py.
Requires GEMINI_API_KEY environment variable.
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

SCORING_INSTRUCTION = (
    "Rate each of the following responses to the question: "
    f'"{PROMPT}"\n\n'
    "For each response, score it on a 1-10 scale considering:\n"
    "- Helpfulness: Does it give useful, practical advice?\n"
    "- Specificity: Does it go beyond generic platitudes?\n"
    "- Actionability: Could someone follow this advice immediately?\n\n"
    "Return a JSON array with one object per response, in the same order."
)


class Response(BaseModel):
    text: str


class ResponseList(BaseModel):
    responses: list[Response]


class Score(BaseModel):
    score: float
    reasoning: str


class ScoreList(BaseModel):
    scores: list[Score]


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable required.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-flash"

    # Step 1: Generate responses
    print(f"Generating responses with {model}...")
    gen_result = client.models.generate_content(
        model=model,
        contents=GENERATION_INSTRUCTION,
        config=genai.types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ResponseList,
        ),
    )
    response_list = json.loads(gen_result.text)
    texts = [r["text"] for r in response_list["responses"]]
    print(f"  Generated {len(texts)} responses.")

    # Step 2: Score all responses in a single call
    print("Scoring responses...")
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    scoring_prompt = f"{SCORING_INSTRUCTION}\n\n{numbered}"

    score_result = client.models.generate_content(
        model=model,
        contents=scoring_prompt,
        config=genai.types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ScoreList,
        ),
    )
    score_list = json.loads(score_result.text)
    scores = score_list["scores"]

    if len(scores) != len(texts):
        print(f"Warning: got {len(scores)} scores for {len(texts)} responses.")
        # Truncate to match
        n = min(len(scores), len(texts))
        texts = texts[:n]
        scores = scores[:n]

    # Normalize scores from 1-10 to [0, 1]
    raw_scores = [s["score"] for s in scores]
    min_s, max_s = min(raw_scores), max(raw_scores)
    if max_s > min_s:
        normalized = [(s - min_s) / (max_s - min_s) for s in raw_scores]
    else:
        normalized = [0.5] * len(raw_scores)

    # Step 3: Save
    output = {
        "prompt": PROMPT,
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

    out_path = os.path.join(os.path.dirname(__file__), "responses.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(texts)} responses to {out_path}")
    print(f"Quality range: {min(normalized):.3f} - {max(normalized):.3f}")


if __name__ == "__main__":
    main()
