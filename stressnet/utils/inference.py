"""Model inference and evaluation helpers for StressNET graph batches.

Typical use is :func:`predict` or :func:`predict_augmented` on dictionaries
produced by preprocessing / data loading, and :func:`frame_with_predicted_tensions`
to attach vector predictions to a ForSys :class:`~forsys.frames.Frame`.
"""

__all__ = ['predict', 'predict_augmented', 'frame_with_predicted_tensions', 'calculate_metrics']

from .data_utils import randomly_rotate_edge_features
import tensorflow as tf
import numpy as np
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from typing import Dict, Any, Literal, TYPE_CHECKING
from copy import deepcopy

if TYPE_CHECKING:
    from forsys.frames import Frame

DEFAULT_DEVICE = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'


def predict(model: tf.keras.Model,
            data: Dict[str, Any],
            device: str | None = None
            ) -> np.ndarray:
    """Run a single forward pass and return flattened stress predictions.

    Parameters
    ----------
    model
        Compiled StressNET Keras model.
    data
        Graph batch with keys ``'x'``, ``'a'`` (sparse adjacency), ``'e'`` (edge features).
    device
        TensorFlow device string. Defaults to GPU when available, else CPU.

    Returns
    -------
    numpy.ndarray
        1-D array of predictions (length equals number of graph nodes for stress mode).
    """
    with tf.device(device or DEFAULT_DEVICE):
        A = sp_matrix_to_sp_tensor(data['a'])
        E = data['e'].reshape(data['e'].shape[0], -1)
        preds = model([data['x'], A, E], training=False).numpy().ravel()
    return preds


def predict_augmented(model: tf.keras.Model,
                      data: Dict[str, Any],
                      n_augmentations: int = 9,
                      seed: int | None = 1337,
                      device: str | None = None,
                      mode: Literal['stress', 'embedding'] = 'stress'
                      ) -> np.ndarray:
    """Average predictions over random rotations of edge features (data augmentation).

    Parameters
    ----------
    model
        Compiled StressNET Keras model.
    data
        Graph batch with keys ``'x'``, ``'a'``, ``'e'``.
    n_augmentations
        Number of *additional* random rotations (total runs = ``n_augmentations + 1``).
    seed
        RNG seed for reproducibility.
    device
        TensorFlow device string.
    mode
        ``'stress'`` rescales the mean prediction to unit mean; ``'embedding'`` returns
        averaged embedding without that rescaling.

    Returns
    -------
    numpy.ndarray
        Mean prediction vector across augmentations.
    """
    rng = np.random.default_rng(seed)
    pred_rows = n_augmentations + 1
    if mode == 'stress':
        # shape = (n_nodes, 1)
        pred_cols = data['x'].shape[0]
        rescale_average_vector = True
    else:
        # shape = (1, embedding_size)
        pred_cols = model.output_shape[1]
        rescale_average_vector = False

    # do n_augmentations + 1 predictions and average results
    preds_array = np.empty((pred_rows, pred_cols), dtype=np.float32)

    # TODO: predict in batch for large n (requires the model to be loaded in disjoint mode)
    with tf.device(device or DEFAULT_DEVICE):
        A = sp_matrix_to_sp_tensor(data['a'])
        E = data['e'].reshape(data['e'].shape[0], -1)
        preds_array[0] = model([data['x'], A, E], training=False).numpy().ravel()
        for i in range(1, pred_rows):
            E_ = randomly_rotate_edge_features(data['e'], rng=rng).reshape(data['e'].shape[0], -1)
            preds_array[i] = model([data['x'], A, E_], training=False).numpy().ravel()
    aug_preds = preds_array.mean(axis=0)

    if rescale_average_vector:
        aug_preds /= aug_preds.mean()

    return aug_preds


def frame_with_predicted_tensions(frame: 'Frame',
                                  predicted_tensions: np.ndarray
                                  ) -> 'Frame':
    """Return a deep-copied frame with predicted tensions assigned to edges.

    Parameters
    ----------
    frame
        Source ForSys frame (unchanged).
    predicted_tensions
        One scalar per *big edge*, in the same order as ``frame.get_big_edges()``.

    Returns
    -------
    forsys.frames.Frame
        New frame instance; each small edge in a big edge receives the same tension.

    Raises
    ------
    ValueError
        If the number of predictions does not match the number of big edges.
    """
    new_frame = deepcopy(frame)
    tensions = np.asarray(predicted_tensions).ravel()
    big_edges = new_frame.get_big_edges()
    if tensions.size != len(big_edges):
        raise ValueError(
            f'Expected {len(big_edges)} tension values (one per big edge), got {tensions.size}.'
        )
    for be, tension in zip(big_edges, tensions):
        for e_id in be.edges:
            new_frame.edges[e_id].tension = float(tension)
    return new_frame


def _borges_score_v2(mape: float,
                     pearsonr: float,
                     r2: float,
                     alpha: float = 1.0,
                     beta: float = 1.0,
                     gamma: float = 1.0,
                     saturate_mape: float = 1.0,
                     saturate_pearsonr: float = 0.99,
                     saturate_r2: float = 0.99
                     ) -> float:
    mape = max(mape, saturate_mape)
    pearsonr = min(pearsonr, saturate_pearsonr)
    r2 = min(r2, saturate_r2)
    return (
        100.0 * alpha / mape
        + (beta / 2.0) * (1.0 + pearsonr) / (1.0 - pearsonr)
        + gamma / (1.0 - r2)
    )


def calculate_metrics(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      ) -> tuple[float, float, float, float]:
    """Correlation, R², MAPE (%), and composite Borges-style score.

    Parameters
    ----------
    y_true
        Ground-truth targets (1-D).
    y_pred
        Model predictions (same shape as ``y_true``).

    Returns
    -------
    tuple[float, float, float, float]
        ``(pearson_r, r2, mape_percent, score)``.
    """
    r = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    score = _borges_score_v2(mape, r, r2)
    return r, r2, mape, score
