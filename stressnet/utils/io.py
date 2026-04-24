__all__ = ['save_graph_data', 'load_graph_data']

import os
import numpy as np
from scipy.sparse import csr_matrix
from logging import getLogger

log = getLogger(__name__)


def save_graph_data(dst_file: str,
                    *,
                    adj_mat: csr_matrix, 
                    node_features: np.ndarray,
                    edge_features: np.ndarray,
                    targets: np.ndarray | None = None,
                    forsys_predictions: np.ndarray | None = None,
                    compressed: bool = False,
                    verbose: bool = False
                    ) -> None:
    """Persist graph tensors (and optional labels) to an ``.npz`` file.

    Parameters
    ----------
    dst_file
        Output path (``.npz``). Parent directories are created as needed.
    adj_mat
        Sparse CSR adjacency matrix.
    node_features
        Dense node feature matrix ``X``.
    edge_features
        Dense edge feature matrix ``E``.
    targets
        Optional ground-truth vector or matrix ``Y``.
    forsys_predictions
        Optional ForSys baseline predictions stored as ``forsys_preds``.
    compressed
        Use ``numpy.savez_compressed`` instead of ``numpy.savez``.
    verbose
        Log a short summary when ``True``.
    """
    extra = {}

    if targets is not None:
        extra['targets'] = targets

    if forsys_predictions is not None:
        extra['forsys_preds'] = forsys_predictions

    if verbose:
        t = " + targets" if targets is not None else ""
        fp = " + forsys predictions" if forsys_predictions is not None else ""
        log.info(f'Saving graph{t}{fp} to "{os.path.normpath(dst_file)}"...')

    # create saving directory
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)

    # save data
    save_method = np.savez if not compressed else np.savez_compressed
    save_method(
        file=dst_file,
        # graph data
        node_features=node_features,
        edge_features=edge_features,
        adj_mat_data=adj_mat.data,
        adj_mat_indices=adj_mat.indices,
        adj_mat_indptr=adj_mat.indptr,
        adj_mat_shape=adj_mat.shape,
        # extra data
        **extra
    )


def load_graph_data(file_path: str,
                    include_forsys_predictions: bool = True,
                    include_targets: bool = True,
                    as_dict: bool = True
                    ):
    """Load graph tensors produced by :func:`save_graph_data`.

    Parameters
    ----------
    file_path
        Path to an ``.npz`` archive written by this module.
    include_forsys_predictions
        If ``False``, omit ``forsys_preds`` even when present in the file.
    include_targets
        If ``False``, omit ``y`` even when present.
    as_dict
        If ``True`` (default), return a dict with keys ``'a'``, ``'x'``, ``'e'``, and
        optionally ``'y'``, ``'forsys_preds'``. If ``False``, return a tuple in fixed order.

    Returns
    -------
    dict or tuple
        Reconstructed adjacency, features, and optional arrays.

    Notes
    -----
    Tuple order when ``as_dict`` is ``False`` is
    ``(A, X, E[, Y][, forsys_predictions])`` depending on the two ``include_*`` flags.
    """
    loader = np.load(file_path)
    adj_mat = csr_matrix(
        (loader['adj_mat_data'], loader['adj_mat_indices'], loader['adj_mat_indptr']),
        shape=loader['adj_mat_shape']
    )
    node_features = loader['node_features']
    edge_features = loader['edge_features']

    out = [adj_mat, node_features, edge_features]

    if include_targets:
        out.append(loader['targets'])

    if include_forsys_predictions:
        out.append(loader['forsys_preds'])

    if as_dict:
        graph_data_keys = ['a', 'x', 'e', 'y', 'forsys_preds']
        return dict(zip(graph_data_keys, out))

    return tuple(out)
