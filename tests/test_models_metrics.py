import numpy as np
import pytest
import torch

from ecg_arrhythmia.metrics import classification_metrics
from ecg_arrhythmia.models import MODEL_FACTORIES, build_model


@pytest.mark.parametrize("model_name", tuple(MODEL_FACTORIES))
def test_model_returns_three_class_logits(model_name):
    model = build_model(model_name)
    logits = model(torch.randn(4, 252))

    assert logits.shape == (4, 3)


def test_abnormal_sensitivity_counts_s_to_v_as_detected_abnormal():
    metrics = classification_metrics(
        true_labels=[0, 1, 1, 2, 2],
        predicted_labels=[0, 2, 0, 2, 1],
    )

    assert metrics["abnormal_sensitivity"] == pytest.approx(0.75)
    assert metrics["per_class"]["S (supraventricular)"]["sensitivity"] == pytest.approx(0)
