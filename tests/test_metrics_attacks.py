from src.attacks import apply_attack, delete_words, substitute_words
from src.metrics import bootstrap_rate_ci, classification_metrics


def test_perfect_classification_metrics():
    result = classification_metrics([0, 0, 1, 1], [0, 0, 1, 1], [-2, -1, 5, 6])
    assert result["accuracy"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["roc_auc_zscore"] == 1.0


def test_bootstrap_interval_for_all_positive():
    low, high = bootstrap_rate_ci([1, 1, 1], iterations=100)
    assert low == 1.0
    assert high == 1.0


def test_attacks_are_deterministic():
    text = "Students need important methods to solve a large problem."
    assert delete_words(text, seed=9) == delete_words(text, seed=9)
    assert substitute_words(text, seed=9) == substitute_words(text, seed=9)
    assert apply_attack(text, "none") == text


def test_substitution_proxy_preserves_basic_text_shape():
    result = substitute_words("This is an important method.", rate=1.0, seed=1)
    assert "significant" in result
