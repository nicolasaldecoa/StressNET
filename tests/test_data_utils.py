import numpy as np
import pandas as pd

from stressnet.utils.data_utils import get_logs_headers, rotate_2d_batch, split_data


def test_get_logs_headers_includes_optional_forsys_and_sorted_params() -> None:
    headers = get_logs_headers({'zeta': 1, 'alpha': 2}, include_forsys_predictions=True)

    assert headers[:5] == ['simulation', 'n_nodes', 'n_edges', 'total_time', 'load_time']
    assert headers[5:8] == ['forsys_mae', 'forsys_mape', 'forsys_pred_time']
    assert headers[-2:] == ['alpha', 'zeta']


def test_get_logs_headers_can_omit_forsys_columns() -> None:
    headers = get_logs_headers({'beta': 1}, include_forsys_predictions=False)

    assert headers == ['simulation', 'n_nodes', 'n_edges', 'total_time', 'load_time', 'beta']


def test_rotate_2d_batch_zero_and_right_angle() -> None:
    points = np.array([[[1.0, 0.0], [0.0, 1.0]]])

    np.testing.assert_allclose(rotate_2d_batch(points, 0.0), points)
    np.testing.assert_allclose(
        rotate_2d_batch(points, np.pi / 2),
        np.array([[[0.0, -1.0], [1.0, 0.0]]]),
        atol=1e-7,
    )


def test_split_data_is_deterministic_for_fixed_seed() -> None:
    dataset = pd.Series([f'sample_{i}' for i in range(10)])

    first = split_data(dataset, val_split=0.2, test_split=0.2, seed=7, as_datasources=False)
    second = split_data(dataset, val_split=0.2, test_split=0.2, seed=7, as_datasources=False)

    assert [len(part) for part in first] == [6, 2, 2]
    assert [part.tolist() for part in first] == [part.tolist() for part in second]
