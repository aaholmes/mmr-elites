"""Tests for examples/llm_response_selection.py."""

import json
import os
import sys
import types
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.spatial.distance import cdist

# Mock sentence_transformers and mmr_elites_rs before importing the module.
_mock_st_module = types.ModuleType("sentence_transformers")
_mock_st_module.SentenceTransformer = MagicMock

_mock_rs = MagicMock()

sys.modules.setdefault("sentence_transformers", _mock_st_module)
sys.modules.setdefault("mmr_elites_rs", _mock_rs)

from examples.llm_response_selection import (  # noqa: E402
    compute_diversity,
    load_responses,
    print_results,
    select_top_k,
)

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_responses_json(tmp_path):
    data = {
        "prompt": "test prompt",
        "model": "test-model",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "responses": [
            {
                "text": "Response about topic %d" % i,
                "quality": i / 9.0,
                "score_reasoning": "ok",
            }
            for i in range(10)
        ],
    }
    path = tmp_path / "responses.json"
    path.write_text(json.dumps(data))
    return str(path)


# ---------------------------------------------------------------------------
# TestLoadResponses
# ---------------------------------------------------------------------------


class TestLoadResponses:
    def test_loads_prompt_and_responses(self, sample_responses_json):
        prompt, responses = load_responses(sample_responses_json)
        assert prompt == "test prompt"
        assert len(responses) == 10

    def test_returns_correct_types(self, sample_responses_json):
        prompt, responses = load_responses(sample_responses_json)
        assert isinstance(prompt, str)
        assert isinstance(responses, list)
        assert isinstance(responses[0], dict)
        assert "text" in responses[0]
        assert "quality" in responses[0]


# ---------------------------------------------------------------------------
# TestSelectTopK
# ---------------------------------------------------------------------------


class TestSelectTopK:
    def test_basic_selection(self):
        quality = np.array([0.1, 0.9, 0.5, 0.3, 0.8])
        indices = select_top_k(quality, 3)
        # Top 3 by quality: indices 1 (0.9), 4 (0.8), 2 (0.5)
        assert set(indices) == {1, 4, 2}

    def test_k_equals_n(self):
        quality = np.array([0.1, 0.9, 0.5])
        indices = select_top_k(quality, 3)
        assert len(indices) == 3
        assert set(indices) == {0, 1, 2}

    def test_returns_descending_quality_order(self):
        quality = np.array([0.3, 0.7, 0.1, 0.9, 0.5])
        indices = select_top_k(quality, 4)
        selected_q = quality[indices]
        # Should be in descending order
        assert all(
            selected_q[i] >= selected_q[i + 1] for i in range(len(selected_q) - 1)
        )


# ---------------------------------------------------------------------------
# TestComputeDiversity
# ---------------------------------------------------------------------------


class TestComputeDiversity:
    def test_identical_vectors_zero_diversity(self):
        emb = np.array([[1.0, 0.0, 0.0]] * 5)
        indices = np.array([0, 1, 2])
        assert compute_diversity(emb, indices) == pytest.approx(0.0)

    def test_orthogonal_vectors_high_diversity(self):
        emb = np.eye(3)  # unit vectors along axes
        indices = np.array([0, 1, 2])
        div = compute_diversity(emb, indices)
        assert div > 0.5  # orthogonal vectors have cosine distance = 1.0

    def test_two_elements(self):
        emb = np.array([[1.0, 0.0], [0.0, 1.0]])
        indices = np.array([0, 1])
        div = compute_diversity(emb, indices)
        expected = cdist(emb[:1], emb[1:], metric="cosine")[0, 0]
        assert div == pytest.approx(expected)


# ---------------------------------------------------------------------------
# TestPrintResults
# ---------------------------------------------------------------------------


class TestPrintResults:
    def _make_args(self, texts=None):
        if texts is None:
            texts = ["Short response", "Another one"]
        responses = [
            {"text": t, "quality": 0.8, "score_reasoning": "ok"} for t in texts
        ]
        quality = np.array([0.8] * len(texts))
        indices = np.arange(len(texts))
        return indices, responses, quality

    def test_output_contains_title(self, capsys):
        indices, responses, quality = self._make_args()
        print_results("Test Title", indices, responses, quality, 0.5)
        captured = capsys.readouterr()
        assert "Test Title" in captured.out

    def test_output_contains_quality_scores(self, capsys):
        indices, responses, quality = self._make_args()
        print_results("T", indices, responses, quality, 0.42)
        captured = capsys.readouterr()
        assert "q=0.80" in captured.out

    def test_truncates_long_text(self, capsys):
        long_text = "A" * 100
        indices, responses, quality = self._make_args([long_text])
        print_results("T", indices, responses, quality, 0.5)
        captured = capsys.readouterr()
        assert "..." in captured.out
        # Should not contain the full 100-char string
        assert "A" * 100 not in captured.out


# ---------------------------------------------------------------------------
# TestMain
# ---------------------------------------------------------------------------


def _make_fake_embeddings(n_responses=10, dim=8):
    """Helper to create normalized fake embeddings."""
    fake = np.random.rand(n_responses, dim).astype(np.float64)
    norms = np.linalg.norm(fake, axis=1, keepdims=True)
    return fake / norms


class TestMain:
    def test_full_pipeline(self, sample_responses_json):
        from examples import llm_response_selection as mod

        fake_embeddings = _make_fake_embeddings()

        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embeddings

        mock_selector = MagicMock()
        mock_selector.select.return_value = np.array([0, 1, 2, 3, 4, 5, 6, 7])

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(mod, "SentenceTransformer", return_value=mock_model)
            )
            mock_rs_mod = stack.enter_context(patch.object(mod, "mmr_elites_rs"))
            mock_rs_mod.MMRSelector.return_value = mock_selector
            mod.main(k=8, lambda_val=0.5, responses_path=sample_responses_json)

    def test_missing_file_exits(self, tmp_path):
        from examples import llm_response_selection as mod

        with pytest.raises(SystemExit):
            mod.main(responses_path=str(tmp_path / "nonexistent.json"))

    def test_custom_responses_path(self, sample_responses_json):
        from examples import llm_response_selection as mod

        fake_embeddings = _make_fake_embeddings()

        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embeddings

        mock_selector = MagicMock()
        mock_selector.select.return_value = np.array([0, 1, 2])

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(mod, "SentenceTransformer", return_value=mock_model)
            )
            mock_rs_mod = stack.enter_context(patch.object(mod, "mmr_elites_rs"))
            mock_rs_mod.MMRSelector.return_value = mock_selector
            mod.main(k=3, lambda_val=0.7, responses_path=sample_responses_json)

    def test_default_responses_path(self, sample_responses_json):
        """Cover the None -> default path branch."""
        from examples import llm_response_selection as mod

        fake_embeddings = _make_fake_embeddings()

        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embeddings

        mock_selector = MagicMock()
        mock_selector.select.return_value = np.array([0, 1, 2, 3, 4, 5, 6, 7])

        # Patch os.path.join to return our fixture path when "responses.json" is involved
        orig_join = os.path.join

        def fake_join(*a):
            if any("responses.json" in str(x) for x in a):
                return sample_responses_json
            return orig_join(*a)

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(mod, "SentenceTransformer", return_value=mock_model)
            )
            mock_rs_mod = stack.enter_context(patch.object(mod, "mmr_elites_rs"))
            stack.enter_context(patch("os.path.join", side_effect=fake_join))
            mock_rs_mod.MMRSelector.return_value = mock_selector
            mod.main()  # no responses_path — exercises the default branch
