from crossmed.metrics import classification_metrics


def test_metrics_perfect_predictions():
    result = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert result["balanced_accuracy"] == 1.0
    assert result["roc_auc"] == 1.0

