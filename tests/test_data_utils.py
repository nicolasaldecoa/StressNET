import numpy as np
import pandas as pd

from stressnet.utils.data_utils import (
    EdgeFeaturesRandomJitter,
    EdgeFeaturesRandomRotation,
    get_logs_headers,
    interpolate_vertices_fill_nans,
    resample_vertices,
    rotate_2d_batch,
    split_data,
)


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


def test_resample_vertices_upsamples_edge_and_preserves_endpoints() -> None:
    incomplete_edge = np.array(
        [
            [0.03160267, 0.56006801],
            [0.03020278, 0.41690132],
            [0.01972519, 0.20849314],
            [0.00733863, 0.06999266],
            [0.0, 0.0],
        ]
    )

    upsampled = resample_vertices(incomplete_edge, target_length=9)

    assert upsampled.shape == (9, 2)
    np.testing.assert_allclose(upsampled[0], incomplete_edge[0])
    np.testing.assert_allclose(upsampled[-1], incomplete_edge[-1], atol=1e-12)
    assert not np.isnan(upsampled).any()


def test_resample_vertices_downsamples_edge_and_preserves_endpoints() -> None:
    edge = np.array(
        [
            [0.03160267, 0.56006801],
            [0.03143293, 0.48816654],
            [0.03020278, 0.41690132],
            [0.02786968, 0.34669667],
            [0.02439123, 0.27738285],
            [0.01972519, 0.20849314],
            [0.01399844, 0.13951865],
            [0.00733863, 0.06999266],
            [0.0, 0.0],
        ]
    )

    downsampled = resample_vertices(edge, target_length=5)

    assert downsampled.shape == (5, 2)
    np.testing.assert_allclose(downsampled[0], edge[0])
    np.testing.assert_allclose(downsampled[-1], edge[-1], atol=1e-12)
    assert not np.isnan(downsampled).any()


def test_interpolate_vertices_fill_nans_restores_missing_edge_points() -> None:
    edge_with_missing_points = np.array(
        [
            [0.0, 1.0],
            [np.nan, np.nan],
            [0.5, 0.5],
            [np.nan, np.nan],
            [1.0, 0.0],
        ]
    )

    interpolated = interpolate_vertices_fill_nans(edge_with_missing_points)

    assert interpolated.shape == edge_with_missing_points.shape
    assert not np.isnan(interpolated).any()
    np.testing.assert_allclose(interpolated[0], [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(interpolated[-1], [1.0, 0.0], atol=1e-12)


def test_rotate_2d_batch_matches_rotating_the_whole_geometry() -> None:
    point_set = np.array(
        [
            [-0.2, -0.3],
            [0.1, 0.1],
            [0.22, 0.33],
            [0.0, 0.50],
            [-0.1, 0.66],
            [0.42, 0.40],
            [0.75, 0.44],
        ]
    )
    junction_idx = 2
    edges_1_2_indices = [0, 1, junction_idx, 3, 4]
    edges_1_3_indices = [0, 1, junction_idx, 5, 6]
    edge_features = np.array(
        [
            point_set[edges_1_2_indices] - point_set[junction_idx],
            point_set[edges_1_3_indices] - point_set[junction_idx],
        ]
    )
    angle = np.pi / 4

    rotated_point_set = rotate_2d_batch(point_set[None, :, :], angle)[0]
    rotated_edge_features = rotate_2d_batch(edge_features, angle)
    expected_edge_features = np.array(
        [
            rotated_point_set[edges_1_2_indices] - rotated_point_set[junction_idx],
            rotated_point_set[edges_1_3_indices] - rotated_point_set[junction_idx],
        ]
    )

    np.testing.assert_allclose(rotated_edge_features, expected_edge_features)


def test_edge_features_random_rotation_is_deterministic_and_preserves_lengths() -> None:
    edge_features = np.array(
        [
            [[0.0, 0.0], [0.2, 0.1], [0.5, 0.5], [0.8, 0.6], [1.0, 1.0]],
            [[0.0, 0.0], [0.1, -0.1], [0.5, -0.5], [0.9, -0.6], [1.0, -1.0]],
        ]
    )
    augmentation = EdgeFeaturesRandomRotation(always_apply=True)

    first = augmentation(edge_features, np.random.default_rng(1337))
    second = augmentation(edge_features, np.random.default_rng(1337))

    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(
        np.linalg.norm(np.diff(first, axis=1), axis=-1),
        np.linalg.norm(np.diff(edge_features, axis=1), axis=-1),
    )


def test_edge_features_random_jitter_is_deterministic_and_preserves_endpoints() -> None:
    edge_features = np.array(
        [
            [[0.0, 0.0], [0.25, 0.1], [0.5, 0.5], [0.75, 0.9], [1.0, 1.0]],
            [[0.0, 0.0], [0.25, -0.2], [0.5, -0.5], [0.75, -0.8], [1.0, -1.0]],
        ]
    )
    augmentation = EdgeFeaturesRandomJitter(jitter_e=0.1, always_apply=True)

    first = augmentation(edge_features, np.random.default_rng(1337))
    second = augmentation(edge_features, np.random.default_rng(1337))

    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first[:, 0, :], edge_features[:, 0, :])
    np.testing.assert_allclose(first[:, -1, :], edge_features[:, -1, :])
    assert not np.allclose(first[:, 1:-1, :], edge_features[:, 1:-1, :])


def test_split_data_is_deterministic_for_fixed_seed() -> None:
    dataset = pd.Series([f'sample_{i}' for i in range(10)])

    first = split_data(dataset, val_split=0.2, test_split=0.2, seed=7, as_datasources=False)
    second = split_data(dataset, val_split=0.2, test_split=0.2, seed=7, as_datasources=False)

    assert [len(part) for part in first] == [6, 2, 2]
    assert [part.tolist() for part in first] == [part.tolist() for part in second]
