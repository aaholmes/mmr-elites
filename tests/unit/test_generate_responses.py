"""Tests for examples/generate_responses.py."""

import json
import os
import sys
import types
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

# Mock google.genai and pydantic before importing the module, since it calls
# sys.exit(1) on ImportError at import time and uses pydantic.BaseModel.
_mock_genai = MagicMock()
_mock_genai.types.GenerateContentConfig = MagicMock

_mock_google = types.ModuleType("google")
_mock_google.genai = _mock_genai

sys.modules.setdefault("google", _mock_google)
sys.modules.setdefault("google.genai", _mock_genai)

# Provide a real-enough BaseModel if pydantic isn't installed (CI environment).
if "pydantic" not in sys.modules:
    _mock_pydantic = types.ModuleType("pydantic")

    class _FakeBaseModel:
        pass

    _mock_pydantic.BaseModel = _FakeBaseModel
    sys.modules["pydantic"] = _mock_pydantic

from examples.generate_responses import (  # noqa: E402
    build_output,
    build_single_scoring_prompt,
    compute_weighted_score,
    generate_responses,
    normalize_scores,
    score_all_responses,
    score_single_response,
)


def _make_score_dict(novelty=6, precision=7, depth=5, completeness=8, reasoning="ok"):
    """Helper to build a score dict matching the new per-criterion schema."""
    return {
        "novelty": novelty,
        "precision": precision,
        "depth": depth,
        "completeness": completeness,
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# TestNormalizeScores
# ---------------------------------------------------------------------------


class TestNormalizeScores:
    def test_basic(self):
        result = normalize_scores([1, 5, 10])
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(4 / 9)
        assert result[2] == pytest.approx(1.0)

    def test_all_same(self):
        assert normalize_scores([5, 5, 5]) == [0.5, 0.5, 0.5]

    def test_single_score(self):
        assert normalize_scores([7]) == [0.5]

    def test_two_scores(self):
        result = normalize_scores([3, 7])
        assert result == [pytest.approx(0.0), pytest.approx(1.0)]

    def test_empty(self):
        assert normalize_scores([]) == []


# ---------------------------------------------------------------------------
# TestComputeWeightedScore
# ---------------------------------------------------------------------------


class TestComputeWeightedScore:
    def test_basic_weights(self):
        scores = {"novelty": 10, "precision": 10, "depth": 10, "completeness": 10}
        assert compute_weighted_score(scores) == pytest.approx(10.0)

    def test_weighted_calculation(self):
        scores = {"novelty": 8, "precision": 6, "depth": 4, "completeness": 2}
        # 8*0.3 + 6*0.3 + 4*0.2 + 2*0.2 = 2.4 + 1.8 + 0.8 + 0.4 = 5.4
        assert compute_weighted_score(scores) == pytest.approx(5.4)

    def test_all_zeros(self):
        scores = {"novelty": 0, "precision": 0, "depth": 0, "completeness": 0}
        assert compute_weighted_score(scores) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestBuildSingleScoringPrompt
# ---------------------------------------------------------------------------


class TestBuildSingleScoringPrompt:
    def test_contains_response_text(self):
        prompt = build_single_scoring_prompt("Use flashcards daily.")
        assert "Use flashcards daily." in prompt

    def test_contains_original_prompt(self):
        prompt = build_single_scoring_prompt("anything")
        assert "fundraising" in prompt

    def test_no_plural_references(self):
        prompt = build_single_scoring_prompt("anything")
        assert "each of the following" not in prompt.lower()
        assert "Return a JSON array" not in prompt

    def test_contains_weighted_criteria(self):
        prompt = build_single_scoring_prompt("anything")
        assert "Novelty" in prompt
        assert "Precision" in prompt
        assert "Depth" in prompt
        assert "Completeness" in prompt
        assert "0.3" in prompt


# ---------------------------------------------------------------------------
# TestScoreSingleResponse
# ---------------------------------------------------------------------------


class TestScoreSingleResponse:
    def _make_client(self, novelty=6, precision=7, depth=5, completeness=8):
        client = MagicMock()
        client.models.generate_content.return_value.text = json.dumps(
            _make_score_dict(novelty, precision, depth, completeness)
        )
        return client

    def test_returns_weighted_score_and_reasoning(self):
        client = self._make_client(8, 6, 4, 2)
        result = score_single_response(client, "model-x", "some response")
        # 8*0.3 + 6*0.3 + 4*0.2 + 2*0.2 = 5.4
        assert result["score"] == pytest.approx(5.4)
        assert result["reasoning"] == "ok"

    def test_returns_per_criterion_scores(self):
        client = self._make_client(8, 6, 4, 2)
        result = score_single_response(client, "model-x", "some response")
        assert result["novelty"] == 8
        assert result["precision"] == 6
        assert result["depth"] == 4
        assert result["completeness"] == 2

    def test_calls_client_with_single_response_prompt(self):
        client = self._make_client()
        score_single_response(client, "model-x", "learn by doing")
        call_args = client.models.generate_content.call_args
        contents = call_args.kwargs.get("contents") or call_args[1].get("contents")
        assert "learn by doing" in contents

    def test_uses_structured_output_schema(self):
        client = self._make_client()
        score_single_response(client, "model-x", "test")
        call_args = client.models.generate_content.call_args
        config = call_args.kwargs.get("config") or call_args[1].get("config")
        assert config is not None


# ---------------------------------------------------------------------------
# TestScoreAllResponses
# ---------------------------------------------------------------------------


def _fake_score_return():
    d = _make_score_dict()
    d["score"] = compute_weighted_score(d)
    return d


class TestScoreAllResponses:
    def test_calls_once_per_response(self):
        with patch(
            "examples.generate_responses.score_single_response",
            return_value=_fake_score_return(),
        ) as mock_score:
            texts = ["a", "b", "c"]
            score_all_responses(MagicMock(), "m", texts)
            assert mock_score.call_count == 3

    def test_returns_list_matching_input_length(self):
        with patch(
            "examples.generate_responses.score_single_response",
            return_value=_fake_score_return(),
        ):
            result = score_all_responses(MagicMock(), "m", ["a", "b", "c", "d"])
            assert len(result) == 4


# ---------------------------------------------------------------------------
# TestGenerateResponses
# ---------------------------------------------------------------------------


class TestGenerateResponses:
    def _make_client(self, texts):
        client = MagicMock()
        response_list = {"responses": [{"text": t} for t in texts]}
        client.models.generate_content.return_value.text = json.dumps(response_list)
        return client

    def test_returns_list_of_strings(self):
        client = self._make_client(["a", "b", "c"])
        result = generate_responses(client, "model-x")
        assert result == ["a", "b", "c"]

    def test_calls_client_with_generation_instruction(self):
        client = self._make_client(["a"])
        generate_responses(client, "model-x")
        call_args = client.models.generate_content.call_args
        contents = call_args.kwargs.get("contents") or call_args[1].get("contents")
        assert "50 pieces of advice" in contents


# ---------------------------------------------------------------------------
# TestBuildOutput
# ---------------------------------------------------------------------------


class TestBuildOutput:
    def _make_output(self):
        texts = ["resp1", "resp2"]
        normalized = [0.0, 1.0]
        scores = [
            _make_score_dict(3, 4, 2, 5, "weak"),
            _make_score_dict(9, 8, 7, 9, "great"),
        ]
        scores[0]["score"] = compute_weighted_score(scores[0])
        scores[1]["score"] = compute_weighted_score(scores[1])
        return build_output("test prompt", "model-x", texts, normalized, scores)

    def test_output_has_required_keys(self):
        output = self._make_output()
        assert "prompt" in output
        assert "model" in output
        assert "generated_at" in output
        assert "responses" in output

    def test_responses_have_quality_and_reasoning(self):
        output = self._make_output()
        for r in output["responses"]:
            assert "text" in r
            assert "quality" in r
            assert "score_reasoning" in r

    def test_responses_have_per_criterion_scores(self):
        output = self._make_output()
        for r in output["responses"]:
            assert "novelty" in r
            assert "precision" in r
            assert "depth" in r
            assert "completeness" in r
            assert "raw_score" in r

    def test_quality_values_are_rounded(self):
        texts = ["a"]
        normalized = [0.33333333]
        s = _make_score_dict()
        s["score"] = compute_weighted_score(s)
        output = build_output("p", "m", texts, normalized, [s])
        assert output["responses"][0]["quality"] == 0.333


# ---------------------------------------------------------------------------
# TestMain
# ---------------------------------------------------------------------------


class TestMain:
    def test_full_pipeline_produces_json(self, tmp_path):
        from examples import generate_responses as mod

        fake_texts = ["resp1", "resp2", "resp3"]
        gen_response = MagicMock()
        gen_response.text = json.dumps({"responses": [{"text": t} for t in fake_texts]})

        def fake_generate_content(**kwargs):
            contents = kwargs.get("contents", "")
            if "50 pieces of advice" in str(contents):
                return gen_response
            # Scoring call — return per-criterion scores
            score_resp = MagicMock()
            score_resp.text = json.dumps(_make_score_dict(7, 6, 5, 8, "good"))
            return score_resp

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = fake_generate_content

        out_path = str(tmp_path / "responses.json")

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}))
            stack.enter_context(
                patch.object(mod.genai, "Client", return_value=mock_client)
            )
            stack.enter_context(
                patch.object(
                    mod.os.path,
                    "join",
                    side_effect=lambda *a: (
                        out_path if "responses.json" in str(a) else os.path.join(*a)
                    ),
                )
            )
            mod.main()

        with open(out_path) as f:
            data = json.load(f)

        assert data["prompt"] == "How should I approach fundraising for my startup?"
        assert len(data["responses"]) == 3
        # Verify per-criterion fields are present
        assert "novelty" in data["responses"][0]
        assert "raw_score" in data["responses"][0]

    def test_missing_api_key_exits(self):
        from examples import generate_responses as mod

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {}, clear=True))
            stack.enter_context(pytest.raises(SystemExit))
            # Ensure GEMINI_API_KEY is absent
            os.environ.pop("GEMINI_API_KEY", None)
            mod.main()
