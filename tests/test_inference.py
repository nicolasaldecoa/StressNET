from contextlib import nullcontext

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import stressnet.utils.inference as inference


class _TensorLike:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def numpy(self):
        return self._values


class _FakeModel:
    output_shape = (None, 2)

    def __init__(self, outputs):
        self._outputs = [np.asarray(output, dtype=np.float32) for output in outputs]
        self.calls = []

    def __call__(self, inputs, training: bool = False):
        self.calls.append({'inputs': inputs, 'training': training})
        output = self._outputs[min(len(self.calls) - 1, len(self._outputs) - 1)]
        return _TensorLike(output)


class _BigEdge:
    def __init__(self, edge_ids):
        self.edges = edge_ids


class _SmallEdge:
    def __init__(self):
        self.tension = None


class _Frame:
    def __init__(self):
        self._big_edges = [_BigEdge([1, 2]), _BigEdge([3])]
        self.edges = {1: _SmallEdge(), 2: _SmallEdge(), 3: _SmallEdge()}

    def get_big_edges(self):
        return self._big_edges


def _graph_data():
    return {
        'x': np.ones((2, 3), dtype=np.float32),
        'a': csr_matrix(np.eye(2, dtype=np.float32)),
        'e': np.arange(8, dtype=np.float32).reshape(2, 2, 2),
    }


def test_predict_uses_flattened_edge_features_and_fake_model(monkeypatch) -> None:
    monkeypatch.setattr(inference.tf, 'device', lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(inference, 'sp_matrix_to_sp_tensor', lambda adj: adj)
    model = _FakeModel(outputs=[np.array([[0.1], [0.2]], dtype=np.float32)])

    preds = inference.predict(model, _graph_data(), device='/CPU:0')

    np.testing.assert_allclose(preds, np.array([0.1, 0.2], dtype=np.float32))
    assert model.calls[0]['training'] is False
    assert model.calls[0]['inputs'][2].shape == (2, 4)


def test_predict_augmented_rescales_stress_predictions(monkeypatch) -> None:
    monkeypatch.setattr(inference.tf, 'device', lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(inference, 'sp_matrix_to_sp_tensor', lambda adj: adj)
    monkeypatch.setattr(inference, 'randomly_rotate_edge_features', lambda edge_features, rng: edge_features)
    model = _FakeModel(outputs=[np.array([2.0, 4.0]), np.array([2.0, 4.0])])

    preds = inference.predict_augmented(model, _graph_data(), n_augmentations=1, mode='stress')

    np.testing.assert_allclose(preds, np.array([2 / 3, 4 / 3], dtype=np.float32))
    assert len(model.calls) == 2


def test_predict_augmented_embedding_mode_does_not_rescale(monkeypatch) -> None:
    monkeypatch.setattr(inference.tf, 'device', lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(inference, 'sp_matrix_to_sp_tensor', lambda adj: adj)
    monkeypatch.setattr(inference, 'randomly_rotate_edge_features', lambda edge_features, rng: edge_features)
    model = _FakeModel(outputs=[np.array([2.0, 4.0]), np.array([4.0, 8.0])])

    preds = inference.predict_augmented(model, _graph_data(), n_augmentations=1, mode='embedding')

    np.testing.assert_allclose(preds, np.array([3.0, 6.0], dtype=np.float32))


def test_calculate_metrics_contract() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    r, r2, mape, score = inference.calculate_metrics(y_true, y_pred)

    assert r == pytest.approx(1.0)
    assert r2 == pytest.approx(1.0)
    assert mape == pytest.approx(0.0)
    assert score == pytest.approx(299.5)


def test_frame_with_predicted_tensions_copies_and_assigns_big_edge_values() -> None:
    frame = _Frame()

    new_frame = inference.frame_with_predicted_tensions(frame, np.array([1.25, 0.75]))

    assert new_frame is not frame
    assert frame.edges[1].tension is None
    assert new_frame.edges[1].tension == pytest.approx(1.25)
    assert new_frame.edges[2].tension == pytest.approx(1.25)
    assert new_frame.edges[3].tension == pytest.approx(0.75)


def test_frame_with_predicted_tensions_validates_length() -> None:
    with pytest.raises(ValueError, match='one per big edge'):
        inference.frame_with_predicted_tensions(_Frame(), np.array([1.0]))
