__all__ = ['ConcatBroadcast', 'NodeRescale', 'StressConv']

import tensorflow as tf
from spektral.layers.convolutional.message_passing import MessagePassing
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Dense, Layer, LayerNormalization


class ConcatBroadcast(Layer):
    """Concatenates a vector per graph (normally the output of a Global Pooling layer)
    to each of the node feature vectors of the respective graphs. Supports "single" and "disjoint" data modes."""

    def build(self, input_shape):
        assert isinstance(input_shape, list), 'inputs must be a list [X, P] or [X, P, I] in disjoint mode'
        assert len(input_shape[0]) == 2, 'X must be a tensor of shape (n_nodes, n_features)'
        if len(input_shape) == 3:
            shape_x, shape_p, shape_i = input_shape
            assert (len(shape_i) == 1) and (shape_i[0] == shape_x[0]), 'I must be a vector of shape (n_nodes, )'
            assert (len(shape_p) == 2), 'P must be a tensor of shape (n_graphs, n_pooled_features) in disjoint mode'
            self.data_mode = 'disjoint'
        elif len(input_shape) == 2:
            shape_x, shape_p = input_shape
            assert shape_p[0] == 1, 'P must be a tensor of shape (1, n_pooled_features) in single mode'
            self.data_mode = 'single'
        else:
            raise NotImplementedError('ConcatBroadcast layer supported data modes are "single" and "disjoint"')
        super().build(input_shape)

    def call(self, inputs):
        if self.data_mode == 'single':
            x, p = inputs  # [(n_nodes, n_features), (1, n_pooled_features)]
            rep_pooled_x = tf.repeat(p, repeats=tf.shape(x)[0], axis=0)  # (n_nodes, n_pooled_features)
            output = K.concatenate([rep_pooled_x, x], axis=-1)  # (n_nodes, n_features + n_pooled_features)
        else:
            x, p, i = inputs  # [(n_nodes, n_features), (n_graphs, n_pooled_features), (n_nodes, )]
            grouped_x = tf.RaggedTensor.from_value_rowids(x, i)  # (n_nodes_graph_0, ... , n_nodes_graph_(n_graphs - 1))
            reps = grouped_x.row_lengths()
            rep_pooled_x = tf.RaggedTensor.from_row_lengths(values=tf.repeat(p, reps, axis=0), row_lengths=reps)
            output = K.concatenate([rep_pooled_x, grouped_x],
                                   axis=-1).values  # (n_nodes, n_features + n_pooled_features)
        return output

    def compute_output_shape(self, input_shape):
        shape_x, shape_p = input_shape[:2]
        return shape_x[0], shape_x[1] + shape_p[1]  # (n_nodes, n_features + n_pooled_features)


class NodeRescale(Layer):
    """Takes a tensor of shape (n_nodes, 1) as input and returns a rescaled tensor of the same shape.
    Supports normalization by "mean" or "max", and "single" and "disjoint" data modes."""
    agg_modes = ('mean', 'max')

    def __init__(self, agg_mode='mean', **kwargs):
        super().__init__(**kwargs)
        assert agg_mode in self.agg_modes, f'Supported values for agg_mode: {self.agg_modes}'
        self.agg_mode = agg_mode

    def build(self, input_shape):
        if isinstance(input_shape, list) and len(input_shape) == 2:
            shape_x, shape_i = input_shape
            assert (len(shape_x) == 2) and (shape_x[1] == 1), 'X must be a tensor of shape (n_nodes, 1)'
            assert (len(shape_i) == 1) and (shape_i[0] == shape_x[0]), 'I must be a vector of shape (n_nodes, )'
            self.data_mode = 'disjoint'
        elif len(input_shape) == 2:
            assert (input_shape[1] == 1), 'X must be a tensor of shape (n_nodes, 1)'
            self.data_mode = 'single'
        else:
            raise NotImplementedError('MeanNorm layer supported data modes are "single" and "disjoint"')
        super().build(input_shape)

    def call(self, inputs):
        if self.data_mode == 'single':  # (n_nodes, 1)
            agg_f = K.mean if (self.agg_mode == 'mean') else K.max
            output = inputs / agg_f(inputs, axis=0)
        else:  # [(n_nodes, 1), (n_nodes, )]
            agg_f = tf.math.segment_mean if (self.agg_mode == 'mean') else tf.math.segment_max
            x, i = inputs
            x = K.squeeze(x, -1)  # (n_nodes, )
            divisor = agg_f(x, i)  # (n_graphs, )
            grouped_x = tf.RaggedTensor.from_value_rowids(x, i)  # (n_nodes_graph_0, ... , n_nodes_graph_(n_graphs - 1))
            res = grouped_x / divisor[:, None]  # (n_nodes_graph_0, ... , n_nodes_graph_(n_graphs - 1))
            output = K.expand_dims(res.values, axis=-1)  # (n_nodes, 1)
        return output

    def compute_output_shape(self, input_shape):
        # (n_nodes, 1) == X.shape
        if self.data_mode == 'single':
            return input_shape
        return input_shape[0]


class StressConv(MessagePassing):
    """
    Layer inspired in CrystalConv by Tian Xie and Jeffrey C. Grossman
    https://graphneural.network/layers/convolution/#:~:text=%5Bsource%5D-,CrystalConv,-spektral.layers.CrystalConv
    """

    def __init__(
            self,
            channels,
            use_layer_norm=True,
            hidden_dim_factor=2,
            aggregate="sum",
            activation="elu",
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs
    ):
        super().__init__(
            aggregate=aggregate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.channels = channels
        self.use_layer_norm = use_layer_norm
        self.hidden_dim_factor = hidden_dim_factor
        self.nonlinearity = activation

    def build(self, input_shape):
        assert len(input_shape) >= 2
        layer_kwargs = {
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_constraint': self.bias_constraint,
            'dtype': self.dtype,
        }
        in_channels = input_shape[0][-1]

        use_bias = not self.use_layer_norm
        self.linear_proj = Dense(self.channels, use_bias=use_bias) if (in_channels != self.channels) else None
        self.dense_a = Dense(self.channels * self.hidden_dim_factor, use_bias=use_bias, **layer_kwargs)

        self.norm = LayerNormalization() if self.use_layer_norm else None
        self.act = Activation(self.nonlinearity) if (self.nonlinearity is not None) else None

        self.dense_b = Dense(self.channels, **layer_kwargs)
        self.gatefunc = Dense(self.channels, **layer_kwargs, activation='sigmoid')

        self.add_norm = LayerNormalization() if self.use_layer_norm else None

        self.built = True

    def message(self, x, e=None):
        x_i = self.get_targets(x)
        x_j = self.get_sources(x)

        z = K.concatenate([x_i, x_j, e], axis=-1)  # (n_edges, n_node_features_(t-1) * 2 + n_edge_features)
        z = self.dense_a(z)
        if self.norm is not None:
            z = self.norm(z)
        if self.act is not None:
            z = self.act(z)

        # output has shape (n_edges, n_node_features_t) and will be aggregated to obtain (n_nodes, n_node_features_t)
        return self.dense_b(z) * self.gatefunc(z)

    def update(self, embeddings, x=None):
        if self.linear_proj is not None:
            x = self.linear_proj(x)
        out = x + embeddings
        if self.add_norm is not None:
            out = self.add_norm(out)
        return out

    @property
    def config(self):
        return {
            'channels': self.channels,
            'use_layer_norm': self.use_layer_norm,
            'hidden_dim_factor': self.hidden_dim_factor
        }
