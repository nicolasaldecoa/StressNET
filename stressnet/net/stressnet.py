__all__ = ['get_model', 'build_stressnet']

from tensorflow.keras.layers import Input, Dense, LayerNormalization, Activation
from spektral.layers import GlobalAttnSumPool
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L2
from .custom_layers import *

DEFAULT_DEVICE = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'


def get_model(weights_path,
              points_per_edge: int = 9, 
              disjoint_mode: bool = False, 
              device: str | None = None, 
              **net_kwargs
              ) -> tf.keras.Model:
    """Build StressNET and optionally load weights on the chosen device.

    Parameters
    ----------
    weights_path
        Path to an ``.h5`` weights file (may be ``None`` only for random init / debugging).
    points_per_edge
        Number of sampled points per edge (must match training / checkpoint).
    disjoint_mode
        If ``True``, build the disjoint-batched variant (Spektral disjoint mode).
    device
        TensorFlow device string; defaults to GPU when available.
    **net_kwargs
        Forwarded to :func:`build_stressnet`.

    Returns
    -------
    tensorflow.keras.Model
        Compiled Keras model.
    """
    tf.keras.backend.clear_session()
    with tf.device(device or DEFAULT_DEVICE):
        model = build_stressnet(
            load_weights_path=weights_path,
            edge_n_vertices=points_per_edge,
            disjoint_mode=disjoint_mode,
            **net_kwargs
        )
    return model


def build_stressnet(load_weights_path: str | None = None,
                    edge_n_vertices: int = 9,
                    disjoint_mode: bool = False,
                    **kwargs
                    ) -> Model:
    """Construct the StressNET graph neural network architecture.

    Parameters
    ----------
    load_weights_path
        Optional path to load trained weights after building.
    edge_n_vertices
        Edge discretization count (controls input edge feature dimension).
    disjoint_mode
        Whether to include disjoint-mode graph index input.
    **kwargs
        Architecture hyperparameters (layer sizes, regularization, ``fine_tune_layers``, etc.).

    Returns
    -------
    tensorflow.keras.Model
        Uncompiled or compiled model per internal ``Model`` factory defaults.
    """
    # = INPUT DIMENSIONS =
    F = kwargs.get('n_node_features', 6)  # number of features per node
    E = 2 * 2 * (edge_n_vertices - 1)  # number of points in edge features times 2 (x and y coordinates)

    # = MODEL HIPERPARAMETERS =
    NON_LINEARITY = kwargs.get('non_linearity', 'elu')
    USE_LAYER_NORM = kwargs.get('use_layer_norm', True)

    N_BLOCKS = kwargs.get('n_blocks', 3)
    EDGE_FEATURES_EXPAND = kwargs.get('edge_features_expand', 512)
    EDGE_FEATURES_SQUEEZE = kwargs.get('edge_features_squeeze', 128)
    X_DIM = kwargs.get('x_dim', (32, 48, 64, 96))
    HIDDEN_DIM_FACTOR = kwargs.get('hidden_dim_factor', 2)
    HEAD_MLP_UNITS = kwargs.get('head_mlp_units', (256, 64))

    EDGES_MLP_REG = L2(kwargs['edges_mlp_l2_reg']) if 'edges_mlp_l2_reg' in kwargs else L2(1e-6)
    CONV_REG = L2(kwargs['conv_l2_reg']) if 'conv_l2_reg' in kwargs else None
    POOL_ATTN_REG = L2(kwargs['global_pool_l2_reg']) if 'global_pool_l2_reg' in kwargs else None
    HEAD_MLP_REG = L2(kwargs['head_mlp_l2_reg']) if 'head_mlp_l2_reg' in kwargs else L2(1e-6)
    OUTPUT_LAYER_REG = L2(kwargs['reg_head_l2_reg']) if 'reg_head_l2_reg' in kwargs else None
    OUTPUT_NORM_MODE = kwargs.get('output_norm_mode', 'mean')

    FINE_TUNE_LAYERS = set(kwargs.get('fine_tune_layers', []))
    assert (not FINE_TUNE_LAYERS) or load_weights_path, 'Freezing layers with randomly initialized weights ??'

    # Model inputs
    x0 = Input(shape=(F,), name='node_feats')
    a = Input((None,), sparse=True, name='adj_mat')
    e0 = Input(shape=(E,), name='edge_feats')
    gi = Input(shape=(), dtype=tf.int32, name='graph_indices') if disjoint_mode else None

    # obtain embeddings from edge geometries

    # expand
    e = Dense(EDGE_FEATURES_EXPAND, kernel_regularizer=EDGES_MLP_REG,
              use_bias=not USE_LAYER_NORM, name=f'edge_feats_fc_1')(e0)
    if USE_LAYER_NORM:
        e = LayerNormalization(name=f'edge_feats_ln_1')(e)
    e = Activation(NON_LINEARITY, name=f'edge_feats_{NON_LINEARITY}_1')(e)
    # squeeze
    e = Dense(EDGE_FEATURES_SQUEEZE, kernel_regularizer=EDGES_MLP_REG,
              use_bias=not USE_LAYER_NORM, name=f'edge_feats_fc_2')(e)
    if USE_LAYER_NORM:
        e = LayerNormalization(name=f'edge_feats_ln_2')(e)

    # graph convolutions
    x = x0
    for i, dim in enumerate(X_DIM, start=1):
        for j in range(1, N_BLOCKS + 1):
            x = StressConv(channels=dim, use_layer_norm=USE_LAYER_NORM, activation=NON_LINEARITY,
                           kernel_regularizer=CONV_REG, hidden_dim_factor=HIDDEN_DIM_FACTOR,
                           name=f'graph_conv_{i}.{j}')([x, a, e])
    x = Activation(NON_LINEARITY, name=f'graph_conv_out_{NON_LINEARITY}')(x)

    # concatenate the pooled representation of all nodes in the grpah to each individual representation
    pool_inputs = [x, gi] if disjoint_mode else x
    pool = GlobalAttnSumPool(attn_kernel_regularizer=POOL_ATTN_REG, name='global_pool')(pool_inputs)

    concat_inputs = [x, pool, gi] if disjoint_mode else [x, pool]
    x = ConcatBroadcast(name='concat_global_feats')(concat_inputs)

    # "head" MLP
    for i, neurons in enumerate(HEAD_MLP_UNITS, start=1):
        x = Dense(neurons, kernel_regularizer=HEAD_MLP_REG, use_bias=not USE_LAYER_NORM, name=f'reg_head_fc_{i}')(x)
        if USE_LAYER_NORM:
            x = LayerNormalization(name=f'reg_head_ln_{i}')(x)
        x = Activation(NON_LINEARITY, name=f'reg_head_{NON_LINEARITY}_{i}')(x)

    # linear regression
    x = Dense(1, name='reg_head_out', kernel_regularizer=OUTPUT_LAYER_REG)(x)

    # rescaling layer (normalize output to have mean equal to 1 and squeeze the last dimension)
    mean_norm_inputs = [x, gi] if disjoint_mode else x
    x = NodeRescale(agg_mode=OUTPUT_NORM_MODE, name=f'{OUTPUT_NORM_MODE}_norm')(mean_norm_inputs)

    # build model object
    model_inputs = [x0, a, e0, gi] if disjoint_mode else [x0, a, e0]
    model = Model(inputs=model_inputs, outputs=x, name='StressNetV0.2')

    # load weights
    if load_weights_path:
        model.load_weights(load_weights_path)

        if FINE_TUNE_LAYERS:
            available_layer_names = {layer.name for layer in model.layers}
            missing_layers = sorted(FINE_TUNE_LAYERS - available_layer_names)
            if missing_layers:
                missing = list(missing_layers)
                raise ValueError(
                    f'Values in fine_tune_layers are not model layer names: {missing}. '
                )
            # freeze all layers except for the ones that we want to train
            for layer in model.layers:
                layer.trainable = (layer.name in FINE_TUNE_LAYERS)

    return model
