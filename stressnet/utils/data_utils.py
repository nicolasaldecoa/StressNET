import os
import warnings
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
from scipy.ndimage import zoom
from spektral.utils import sp_matrix_to_sp_tensor

from .io import load_graph_data

warnings.simplefilter(action='ignore', category=FutureWarning)

__all__ = ['split_data', 'load_npz_paths_from_logs', 'load_split_merge_from_logs', 'GraphDataGenerator',
           'rotate_2d_batch', 'get_logs_headers', 'ConnectedNodes', 'DisjointModeDataGenerator',
           'interpolate_vertices_fill_nans', 'resample_vertices', 'EdgeFeaturesRandomRotation',
           'EdgeFeaturesRandomJitter']

HEADERS_BASE = ('simulation', 'n_nodes', 'n_edges', 'total_time', 'load_time')
HEADERS_FS = ('forsys_mae', 'forsys_mape', 'forsys_pred_time')


def get_logs_headers(params_dict: dict, include_forsys_predictions: bool = True):
    headers = list(HEADERS_BASE)
    if include_forsys_predictions:
        headers.extend(list(HEADERS_FS))
    headers += sorted(params_dict.keys())
    return headers


def rotate_2d_batch(points_batch, angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[c, s],
                    [-s, c]])
    return np.einsum('ijk,lk->ijl', points_batch, rot)


def randomly_rotate_edge_features(edge_features, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    random_angle = 2 * np.pi * rng.random()  # [0.0, ~6.2831)
    return rotate_2d_batch(edge_features, random_angle)


def randomly_jitter_edge_features(points_batch, rng=None, e=0.05):
    if rng is None:
        rng = np.random.default_rng()
    deltas = np.diff(points_batch, axis=1)
    deltas[:, 1::2, :] *= -1
    spans = np.abs(np.diff(deltas, axis=1))  # spans[h,i,j] == distance between points[h,i-1,j] and points[h,i+1,j]
    noise = rng.normal(np.zeros_like(spans), spans * e)  # std as a fraction of each span
    ret = points_batch.copy()
    ret[:, 1:-1, :] += noise  # don't alter junction points
    return ret


def load_npz_paths_from_logs(logs_csv_path: str):
    return pd.read_csv(logs_csv_path).dropna(subset='n_nodes')['simulation'].reset_index(drop=True)


def split_data(dataset: pd.Series, val_split=0.1, test_split=0.2, subsample=1.0, seed=1337, as_datasources=True):
    kwarg = {'n': subsample} if (subsample > 1) else {'frac': subsample}
    ds = dataset.sample(**kwarg, random_state=seed)
    vs = len(ds) - test_split if (test_split > 1) else int((1 - test_split) * len(ds))
    ts = len(ds) - test_split - val_split if (val_split > 1) else int((1 - test_split - val_split) * len(ds))
    train, val, test = ds.iloc[:ts], ds.iloc[ts:vs], ds.iloc[vs:]
    if as_datasources:
        return DataSources(train, val, test)
    return train, val, test


def load_split_merge_from_logs(logs_csv_paths: list, val_split=0.2, test_split=0.2, subsample=1.0, seed=1337,
                               as_datasources=True):
    train, val, test = [pd.Series(dtype=str) for _ in range(3)]
    for csv in logs_csv_paths:
        series = load_npz_paths_from_logs(csv)
        tr, v, te = split_data(series, val_split, test_split, subsample, seed, as_datasources=False)
        train = pd.concat([train, tr], ignore_index=True)
        val = pd.concat([val, v], ignore_index=True)
        test = pd.concat([test, te], ignore_index=True)
    if as_datasources:
        return DataSources(train, val, test)
    return train, val, test


def get_fs_to_stressnet_bigedge_idx_mapping(targets, removed_nodes):
    if len(removed_nodes) == 0:
        return {i: i for i in range(len(targets))}
    raise NotImplementedError()
    # TODO: write this func


def interpolate_vertices_fill_nans(array_with_nans: np.ndarray, spline_order=2):
    nans = np.isnan(array_with_nans[:, 0])
    assert nans.any(), f'No NaNs in input array: {array_with_nans}'
    real_points = array_with_nans[~nans]
    return zoom(real_points, (len(array_with_nans) / len(real_points), 1), order=spline_order)


def resample_vertices(orig_array: np.ndarray, target_length: int, spline_order=2):
    return zoom(orig_array, (target_length / len(orig_array), 1), order=spline_order)


def to_disjoint(x_list, a_list, e_list, y_list=None):
    """Adapted from spektral.data.utils.to_disjoint"""
    # Node features (x)
    x_out = np.vstack(x_list)
    # Adjacency matrix (a)
    a_out = sp.block_diag(a_list)
    # Edge attributes (e)
    e_out = np.vstack(e_list)
    # Batch indices (i)
    n_nodes = np.array([x.shape[0] for x in x_list])
    i_out = np.repeat(np.arange(len(n_nodes)), n_nodes)
    # [Targets (y)]
    if y_list is not None:
        y_out = np.vstack(y_list)
        return x_out, a_out, e_out, i_out, y_out
    return x_out, a_out, e_out, i_out


# == CLASSES ==


class Augmentation(ABC):
    def __init__(self, p: float = 0.5, always_apply: bool = False):
        assert 0 < p <= 1, 'p must be a float in the range (0, 1]'
        self.p = None if (always_apply or p == 1) else p

    def __call__(self, edge_features: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if (self.p is None) or (rng.random() <= self.p):
            return self._transform(edge_features, rng)
        return edge_features

    @abstractmethod
    def _transform(self, edge_features: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        pass


class EdgeFeaturesRandomRotation(Augmentation):
    def __init__(self, p: float = 0.5, always_apply: bool = False):
        super().__init__(p, always_apply)

    def _transform(self, edge_features: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return randomly_rotate_edge_features(edge_features, rng)


class EdgeFeaturesRandomJitter(Augmentation):
    def __init__(self, jitter_e: float = 0.05, p: float = 0.5, always_apply: bool = False):
        super().__init__(p, always_apply)
        self.e = jitter_e

    def _transform(self, edge_features: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return randomly_jitter_edge_features(edge_features, rng, self.e)


class DataSources:
    def __init__(self, train_set: pd.Series, validation_set: pd.Series, test_set: pd.Series):
        self._train = train_set
        self._val = validation_set
        self._test = test_set

    @property
    def train(self):
        return self._train.copy()

    @property
    def val(self):
        return self._val.copy()

    @property
    def test(self):
        return self._test.copy()

    @property
    def n_train(self):
        return len(self._train)

    @property
    def n_val(self):
        return len(self._val)

    @property
    def n_test(self):
        return len(self._test)

    def to_csv(self, output_path: str):
        splits = pd.DataFrame({
            'simulation': pd.concat([self._train, self._val, self._test], ignore_index=True),
            'split': ['train'] * self.n_train + ['validation'] * self.n_val + ['test'] * self.n_test
        })
        splits.to_csv(output_path, index=False)

    def __repr__(self):
        return f'DataSources(n_train={self.n_train},n_val={self.n_val},n_test={self.n_test})'


class ConnectedNodes:
    """Assumes bidirectional connections"""

    def __init__(self, adj_matrix: sp.csr_matrix):
        self._mask = (np.diff(adj_matrix.indptr) != 0)
        # the rest is done lazily

    def get_boolean_mask(self, which: Literal['connected', 'disconnected'] = 'connected') -> np.ndarray:
        return self._mask.copy() if (which == 'connected') else ~self._mask

    def get_indices(self, which: Literal['connected', 'disconnected'] = 'connected') -> np.ndarray:
        mask = self._mask if (which == 'connected') else ~self._mask
        return np.nonzero(mask)[0]

    def get_count(self, which: Literal['connected', 'disconnected'] = 'connected') -> int:
        count = self._mask.sum() if (which == 'connected') else (self._mask.shape[0] - self._mask.sum())
        return int(count)

    def assert_all_connected(self) -> None:
        assert np.all(self._mask), f'Found disconnected nodes: {np.nonzero(~self._mask)[0].tolist()}'


class GraphDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_series: pd.Series, basedir: str = './', shuffle: bool = True, e_augmentations: list = None,
                 validate_all_nodes_in_graph: bool = False, flatten_edge_feats: bool = True,
                 only_compute_loss_for: pd.Series = None, rescale_y: Literal['mean', 'max'] = 'mean',
                 random_seed: int = None):
        self.data_series = data_series.copy()
        self.shuffle = shuffle
        self.e_augmentations = e_augmentations or []
        self.basedir = basedir
        self.validate = validate_all_nodes_in_graph
        self.flatten_e = flatten_edge_feats
        assert rescale_y in {'mean', 'max'}
        self.rescale_y = rescale_y  # we assume it is usually scaled by mean, so if this is true we do max rescaling
        self.rng = np.random.default_rng(random_seed)

        if only_compute_loss_for is not None:
            assert (only_compute_loss_for.index == self.data_series.index).all()
            self.only_include = only_compute_loss_for.copy()
        else:
            self.only_include = None

        self._set_n_steps()
        self.on_epoch_end()

    def __len__(self):
        """Number of batches in the sequence"""
        return self.n_steps

    def __getitem__(self, index):
        """Yield a single graph and target values for each node"""
        x, a, e, y = self._load_graph(index)
        a = sp_matrix_to_sp_tensor(a)
        if self.only_include is None:
            return [x, a, e], y
        w = self._compute_sample_weights(y, index)
        return [x, a, e], y, w

    def on_epoch_end(self):
        if self.shuffle:
            indices = self.rng.permutation(self.data_series.index)
            self.data_series = self._reindex_series(self.data_series, indices)
            if self.only_include is not None:
                self.only_include = self._reindex_series(self.only_include, indices)

    def _compute_sample_weights(self, target, index):
        """Assign a weight of 1 to the nodes to include and 0 to all other nodes."""
        weights = np.zeros(shape=(target.shape[0], 1), dtype=np.float32)  # spurious dimension or keras fails
        weights[self.only_include.iloc[index]] = [1]
        # this will be divided by the total length of the vector, including masked ones, so we should rescale the vector
        weights *= (weights.shape[0] / weights.sum())
        return weights

    def _set_n_steps(self):
        self.n_steps = len(self.data_series)

    @staticmethod
    def _reindex_series(series: pd.Series, indices: np.ndarray):
        return series.reindex(indices).reset_index(drop=True)

    @staticmethod
    def _assert_no_disconnected_nodes(a, x):
        """Raise an exception if the number of rows in the node features matrix
        doesn't match the number of rows of the adjacency matrix, or if the adjacency
        matrix has any empty rows (assumes that all connections are bidirectional)"""
        ConnectedNodes(a).assert_all_connected()
        assert a.shape[0] == x.shape[0], f'{a.shape[0]} rows in A and {x.shape[0]} rows in X.'

    def _load_graph(self, index):
        filepath = os.path.join(self.basedir, self.data_series.iloc[index])
        a, x, e, y = load_graph_data(filepath, include_forsys_predictions=False, as_dict=False)
        if self.validate:
            self._assert_no_disconnected_nodes(a, x)
        for aug in self.e_augmentations:
            e = aug(e, self.rng)
        if self.flatten_e:
            e = e.reshape(e.shape[0], -1)
        if self.rescale_y == 'max':
            y /= y.max()
        return x, a, e, y.reshape(-1, 1)  # spurious dimension in targets to avoid issues with some keras methods


class DisjointModeDataGenerator(GraphDataGenerator):
    def __init__(self, data_series: pd.Series, batch_size: int = 8, basedir: str = './', shuffle: bool = True,
                 e_augmentations: list = None, validate_all_nodes_in_graph: bool = False,
                 flatten_edge_feats: bool = True, only_compute_loss_for: pd.Series = None, rescale_y: str = 'mean',
                 random_seed: int = None):
        if only_compute_loss_for is not None:
            # TODO: implement only_compute_loss_for in disjoint mode
            raise NotImplementedError('parameter `only_compute_loss_for` not supported in DisjointModeDataGenerator')
        self.batch_size = batch_size
        super().__init__(data_series, basedir, shuffle, e_augmentations,
                                                        validate_all_nodes_in_graph, flatten_edge_feats,
                                                        only_compute_loss_for, rescale_y, random_seed)

    def _set_n_steps(self):
        self.n_steps = int(len(self.data_series) / self.batch_size)  # drop last batch

    def __getitem__(self, index):
        """Yield a batch of graphs in Spektral's disjoint mode and stacked target values for all nodes"""
        start = index * self.batch_size
        end = start + self.batch_size
        graphs = [self._load_graph(idx) for idx in range(start, end)]
        x_tup, a_tup, e_tup, y_tup = zip(*graphs, strict=True)
        x, a, e, i, y = to_disjoint(x_tup, a_tup, e_tup, y_tup)
        a = sp_matrix_to_sp_tensor(a)
        return [x, a, e, i], y
