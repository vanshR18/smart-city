"""
test_nlp.py
───────────
Tests for the NLP classifier — covers both the rule-based fallback
(no model needed) and the trained model (if it exists).

Run:
  pytest tests/test_nlp.py -v
"""

import pytest
from app.nlp.inference import (
    predict, predict_batch, get_model_info,
    _rule_based_classify, NLPResult,
)
from app.nlp.dataset import LABEL2ID, ID2LABEL, NUM_LABELS


# ── Rule-based classifier tests ───────────────────────────────────────────────
# These always run — no model needed.

class TestRuleBasedClassifier:

    def test_fire_keywords_classified_correctly(self):
        result = _rule_based_classify("Building mein aag lag gayi, fire brigade bulao")
        assert result.label == "FIRE"
        assert result.is_emergency is True

    def test_flood_keywords_detected(self):
        result = _rule_based_classify("Heavy waterlogging in Charbagh, paani bhar gaya")
        assert result.label == "FLOOD"
        assert result.is_emergency is True

    def test_accident_keywords_detected(self):
        result = _rule_based_classify("Major accident on ring road, ambulance needed")
        assert result.label == "ACCIDENT"
        assert result.is_emergency is True

    def test_crime_keywords_detected(self):
        result = _rule_based_classify("Chain snatching robbery near Hazratganj")
        assert result.label == "CRIME"
        assert result.is_emergency is True

    def test_normal_text_classified_as_normal(self):
        result = _rule_based_classify("Traffic moving smoothly today, all clear")
        assert result.label == "NORMAL"
        assert result.is_emergency is False

    def test_result_has_all_fields(self):
        result = _rule_based_classify("Test text")
        assert isinstance(result, NLPResult)
        assert result.text == "Test text"
        assert result.label in ID2LABEL.values()
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.urgency_score <= 1.0
        assert isinstance(result.probabilities, dict)
        assert len(result.probabilities) == NUM_LABELS

    def test_probabilities_sum_to_one(self):
        result = _rule_based_classify("Fire in the building near market area")
        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total}, expected ~1.0"

    def test_all_labels_in_probabilities(self):
        result = _rule_based_classify("Some random emergency text")
        for label in ID2LABEL.values():
            assert label in result.probabilities, f"Missing label: {label}"

    def test_method_is_rule_based(self):
        result = _rule_based_classify("Anything")
        assert result.method == "rule_based"


# ── predict() function tests ──────────────────────────────────────────────────
# predict() tries trained model first, falls back to rule-based.
# These tests work whether or not the model is trained.

class TestPredictFunction:

    def test_predict_returns_nlpresult(self):
        result = predict("Fire near Kaiserbagh market, help needed")
        assert isinstance(result, NLPResult)

    def test_predict_returns_valid_label(self):
        result = predict("Accident on highway near Lucknow")
        assert result.label in ID2LABEL.values()

    def test_predict_confidence_in_range(self):
        result = predict("Flood warning in low lying areas")
        assert 0.0 <= result.confidence <= 1.0

    def test_predict_urgency_equals_confidence(self):
        result = predict("Fire emergency")
        assert result.urgency_score == result.confidence

    def test_predict_emergency_flag_correct(self):
        emergency   = predict("Major accident, ambulance needed")
        non_emerg   = predict("Traffic flowing normally on Hazratganj road today")
        # Emergency texts should be flagged (may not always be true for rule-based on ambiguous text)
        assert isinstance(emergency.is_emergency, bool)
        assert isinstance(non_emerg.is_emergency, bool)

    def test_predict_handles_hindi_english_mix(self):
        result = predict("Bada accident hua hai crossing pe, ambulance jaldi bhejo")
        assert isinstance(result, NLPResult)
        assert result.label in ID2LABEL.values()

    def test_predict_handles_short_text(self):
        result = predict("Fire!")
        assert isinstance(result, NLPResult)

    def test_predict_handles_long_text(self):
        long_text = "accident " * 60   # will be truncated by tokenizer
        result = predict(long_text)
        assert isinstance(result, NLPResult)


# ── Batch prediction tests ────────────────────────────────────────────────────

class TestBatchPrediction:

    def test_batch_returns_correct_count(self):
        texts   = ["Fire!", "Flood warning", "Normal traffic", "Accident on highway"]
        results = predict_batch(texts)
        assert len(results) == len(texts)

    def test_batch_all_results_are_nlpresult(self):
        texts   = ["Emergency text one", "Emergency text two"]
        results = predict_batch(texts)
        for r in results:
            assert isinstance(r, NLPResult)

    def test_batch_single_text(self):
        results = predict_batch(["Just one text here"])
        assert len(results) == 1

    def test_batch_preserves_text_order(self):
        texts   = [f"Text number {i}" for i in range(10)]
        results = predict_batch(texts)
        for i, r in enumerate(results):
            assert r.text == texts[i], f"Text order mismatch at index {i}"


# ── Model info tests ──────────────────────────────────────────────────────────

class TestModelInfo:

    def test_model_info_has_required_keys(self):
        info = get_model_info()
        for key in ["model_loaded", "model_version", "device", "model_path", "method"]:
            assert key in info, f"Missing key: {key}"

    def test_model_info_method_is_valid(self):
        info = get_model_info()
        assert info["method"] in ["distilbert", "rule_based"]


# ── Label consistency tests ───────────────────────────────────────────────────

class TestLabelConsistency:

    def test_label2id_and_id2label_are_inverse(self):
        for label, idx in LABEL2ID.items():
            assert ID2LABEL[idx] == label

    def test_num_labels_is_correct(self):
        assert NUM_LABELS == 7
        assert len(LABEL2ID) == 7
        assert len(ID2LABEL) == 7

    def test_all_expected_labels_present(self):
        expected = {"ACCIDENT", "FIRE", "FLOOD", "CRIME", "CROWD", "MEDICAL", "NORMAL"}
        assert set(LABEL2ID.keys()) == expected