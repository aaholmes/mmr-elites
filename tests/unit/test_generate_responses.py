"""Tests for examples/generate_responses.py."""

import json
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# Mock google.genai before importing the module, since it calls sys.exit(1)
# on ImportError at import time.
_mock_genai = MagicMock()
_mock_genai.types.GenerateContentConfig = MagicMock

_mock_google = types.ModuleType("google")
_mock_google.genai = _mock_genai

sys.modules.setdefault("google", _mock_google)
sys.modules.setdefault("google.genai", _mock_genai)

from examples.generate_responses import (  # noqa: E402
    build_output,
    build_single_scoring_prompt,
    generate_responses,
    normalize_scores,
    score_all_responses,
    score_single_response,
)


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
# TestBuildSingleScoringPrompt
# ---------------------------------------------------------------------------


class TestBuildSingleScoringPrompt:
    def test_contains_response_text(self):
        prompt = build_single_scoring_prompt("Use flashcards daily.")
        assert "Use flashcards daily." in prompt

    def test_contains_original_prompt(self):
        prompt = build_single_scoring_prompt("anything")
        assert "What strategies help someone learn to code?" in prompt

    def test_no_plural_references(self):
        prompt = build_single_scoring_prompt("anything")
        assert "each of the following" not in prompt.lower()
        # No numbered-list instruction
        assert "Return a JSON array" not in prompt


# ---------------------------------------------------------------------------
# TestScoreSingleResponse
# ---------------------------------------------------------------------------


class TestScoreSingleResponse:
    def _make_client(self, score=7.5, reasoning="solid advice"):
        client = MagicMock()
        client.models.generate_content.return_value.text = json.dumps(
            {"score": score, "reasoning": reasoning}
        )
        return client

    def test_returns_score_and_reasoning(self):
        client = self._make_client(8.0, "very helpful")
        result = score_single_response(client, "model-x", "some response")
        assert result["score"] == 8.0
        assert result["reasoning"] == "very helpful"

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
        # Config was constructed — just verify it was passed
        assert config is not None


# ---------------------------------------------------------------------------
# TestScoreAllResponses
# ---------------------------------------------------------------------------


class TestScoreAllResponses:
    def test_calls_once_per_response(self):
        with patch(
            "examples.generate_responses.score_single_response",
            return_value={"score": 5.0, "reasoning": "ok"},
        ) as mock_score:
            texts = ["a", "b", "c"]
            score_all_responses(MagicMock(), "m", texts)
            assert mock_score.call_count == 3

    def test_returns_list_matching_input_length(self):
        with patch(
            "examples.generate_responses.score_single_response",
            return_value={"score": 5.0, "reasoning": "ok"},
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
        assert "25 diverse strategies" in contents


# ---------------------------------------------------------------------------
# TestBuildOutput
# ---------------------------------------------------------------------------


class TestBuildOutput:
    def _make_output(self):
        texts = ["resp1", "resp2"]
        normalized = [0.0, 1.0]
        scores = [
            {"score": 3, "reasoning": "weak"},
            {"score": 9, "reasoning": "great"},
        ]
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

    def test_quality_values_are_rounded(self):
        texts = ["a"]
        normalized = [0.33333333]
        scores = [{"score": 5, "reasoning": "ok"}]
        output = build_output("p", "m", texts, normalized, scores)
        assert output["responses"][0]["quality"] == 0.333


# ---------------------------------------------------------------------------
# TestMain
# ---------------------------------------------------------------------------


class TestMain:
    def test_full_pipeline_produces_json(self, tmp_path):
        from examples import generate_responses as mod

        fake_texts = ["resp1", "resp2", "resp3"]
        gen_response = MagicMock()
        gen_response.text = json.dumps(
            {"responses": [{"text": t} for t in fake_texts]}
        )

        def fake_generate_content(**kwargs):
            contents = kwargs.get("contents", "")
            if "25 diverse strategies" in str(contents):
                return gen_response
            # Scoring call — return a single score
            score_resp = MagicMock()
            score_resp.text = json.dumps({"score": 7.0, "reasoning": "good"})
            return score_resp

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = fake_generate_content

        out_path = str(tmp_path / "responses.json")

        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch.object(mod.genai, "Client", return_value=mock_client),
            patch.object(
                mod.os.path,
                "join",
                side_effect=lambda *a: out_path
                if "responses.json" in str(a)
                else os.path.join(*a),
            ),
        ):
            mod.main()

        with open(out_path) as f:
            data = json.load(f)

        assert data["prompt"] == "What strategies help someone learn to code?"
        assert len(data["responses"]) == 3

    def test_missing_api_key_exits(self):
        from examples import generate_responses as mod

        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(SystemExit),
        ):
            # Ensure GEMINI_API_KEY is absent
            os.environ.pop("GEMINI_API_KEY", None)
            mod.main()
