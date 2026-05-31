import logging

import numpy as np
from scipy.sparse import csr_matrix

from stressnet.utils.io import load_graph_data, save_graph_data


def _tiny_graph():
    adj_mat = csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32))
    node_features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    edge_features = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    targets = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    forsys_predictions = np.array([0.6, 0.9, 1.4], dtype=np.float32)
    return adj_mat, node_features, edge_features, targets, forsys_predictions


def test_save_and_load_graph_data_roundtrip_as_dict(tmp_path) -> None:
    adj_mat, node_features, edge_features, targets, forsys_predictions = _tiny_graph()
    path = tmp_path / 'graph.npz'

    save_graph_data(
        str(path),
        adj_mat=adj_mat,
        node_features=node_features,
        edge_features=edge_features,
        targets=targets,
        forsys_predictions=forsys_predictions,
        compressed=True,
    )

    loaded = load_graph_data(str(path), as_dict=True)

    np.testing.assert_array_equal(loaded['a'].toarray(), adj_mat.toarray())
    np.testing.assert_array_equal(loaded['x'], node_features)
    np.testing.assert_array_equal(loaded['e'], edge_features)
    np.testing.assert_array_equal(loaded['y'], targets)
    np.testing.assert_array_equal(loaded['forsys_preds'], forsys_predictions)


def test_load_graph_data_tuple_order_and_optional_fields(tmp_path) -> None:
    adj_mat, node_features, edge_features, targets, forsys_predictions = _tiny_graph()
    path = tmp_path / 'graph.npz'
    save_graph_data(
        str(path),
        adj_mat=adj_mat,
        node_features=node_features,
        edge_features=edge_features,
        targets=targets,
        forsys_predictions=forsys_predictions,
    )

    loaded = load_graph_data(
        str(path),
        include_targets=False,
        include_forsys_predictions=False,
        as_dict=False,
    )

    assert len(loaded) == 3
    np.testing.assert_array_equal(loaded[0].toarray(), adj_mat.toarray())
    np.testing.assert_array_equal(loaded[1], node_features)
    np.testing.assert_array_equal(loaded[2], edge_features)


def test_save_graph_data_verbose_mentions_forsys_predictions(tmp_path, caplog) -> None:
    adj_mat, node_features, edge_features, _targets, forsys_predictions = _tiny_graph()
    caplog.set_level(logging.INFO, logger='stressnet.utils.io')

    save_graph_data(
        str(tmp_path / 'graph.npz'),
        adj_mat=adj_mat,
        node_features=node_features,
        edge_features=edge_features,
        forsys_predictions=forsys_predictions,
        verbose=True,
    )

    assert 'forsys predictions' in caplog.text
    assert 'targets' not in caplog.text
