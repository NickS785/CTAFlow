"""
TensorFlow/Keras implementations of encoder modules.

Migrated from CTAFlow.models.deep_learning.encoders (PyTorch).
"""

import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ----------------------------
# Small blocks
# ----------------------------


class AttnPool(layers.Layer):
    """Attention pooling over time: (B, T, D) -> (B, D)"""

    def __init__(self, d: int, **kwargs):
        super().__init__(**kwargs)
        self.d = d
        self.score = layers.Dense(1)

    def call(self, x, mask=None, training=None):
        # x: (B, T, D), mask: (B, T) 1=valid, 0=pad
        logits = tf.squeeze(self.score(x), axis=-1)  # (B, T)
        if mask is not None:
            logits = tf.where(mask == 0, tf.constant(-1e9, dtype=logits.dtype), logits)
        w = tf.nn.softmax(logits, axis=-1)  # (B, T)
        return tf.einsum("bt,btd->bd", w, x)

    def get_config(self):
        config = super().get_config()
        config.update({"d": self.d})
        return config


class GatedFusion(layers.Layer):
    """Learns per-sample weights for [z_profile, z_seq, z_sum]."""

    def __init__(self, d: int, n_mod: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.d = d
        self.n_mod = n_mod
        self.gate = keras.Sequential([
            layers.Dense(d, activation="gelu"),
            layers.Dense(n_mod),
        ])

    def call(self, zs, training=None):
        # zs: list of (B, D)
        cat = tf.concat(zs, axis=-1)  # (B, n_mod*D)
        w = tf.nn.softmax(self.gate(cat), axis=-1)  # (B, n_mod)
        z = tf.zeros_like(zs[0])
        for i, zi in enumerate(zs):
            z = z + zi * w[:, i:i + 1]
        return z, w

    def get_config(self):
        config = super().get_config()
        config.update({"d": self.d, "n_mod": self.n_mod})
        return config


class SpatialFuse(layers.Layer):
    """Fuses profile and raster/NumberBars encodings with gated or mean mode."""

    def __init__(self, d_spatial: int, mode: str = "gated", temperature: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.d_spatial = d_spatial
        self.mode = mode
        self.temperature = temperature
        self.last_importance = None

        if mode == "gated":
            self.gate_norm = layers.LayerNormalization()
            self.gate = keras.Sequential([
                layers.Dense(d_spatial, activation="gelu"),
                layers.Dense(2),
            ])
        elif mode != "mean":
            raise ValueError(f"Unknown SpatialFuse mode: {mode}")

    def call(self, z_profile, z_nb=None, return_importance=False, training=None):
        if z_nb is None:
            if return_importance:
                batch_size = tf.shape(z_profile)[0]
                importance = tf.broadcast_to(
                    tf.constant([[1.0, 0.0]]),
                    [batch_size, 2]
                )
                self.last_importance = importance
                return z_profile, importance
            return z_profile

        if self.mode == "mean":
            fused = 0.5 * (z_profile + z_nb)
            batch_size = tf.shape(z_profile)[0]
            self.last_importance = tf.broadcast_to(
                tf.constant([[0.5, 0.5]]),
                [batch_size, 2]
            )
            return (fused, self.last_importance) if return_importance else fused

        # Gated mode
        cat_input = tf.concat([z_profile, z_nb], axis=-1)
        gate_input = self.gate_norm(cat_input)
        logits = self.gate(gate_input)
        weights = tf.nn.softmax(logits / self.temperature, axis=-1)
        self.last_importance = tf.stop_gradient(weights)
        fused = weights[:, 0:1] * z_profile + weights[:, 1:2] * z_nb

        if return_importance:
            return fused, weights
        return fused

    def get_importance_stats(self):
        """Get statistics about modality importance from last forward pass."""
        if self.last_importance is None:
            return None

        profile_weights = self.last_importance[:, 0]
        nb_weights = self.last_importance[:, 1]

        return {
            'profile_mean': float(tf.reduce_mean(profile_weights)),
            'nb_mean': float(tf.reduce_mean(nb_weights)),
            'profile_std': float(tf.math.reduce_std(profile_weights)),
            'nb_std': float(tf.math.reduce_std(nb_weights)),
            'profile_dominant': float(tf.reduce_mean(tf.cast(profile_weights > nb_weights, tf.float32))),
        }

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_spatial": self.d_spatial,
            "mode": self.mode,
            "temperature": self.temperature,
        })
        return config


# ----------------------------
# Encoders
# ----------------------------


class ProfileEncoder(layers.Layer):
    """
    Profile encoder using CNN architecture.

    Input: (B, num_bins, in_ch) -> Output: (B, d_out)
    Note: TensorFlow uses channels_last by default, so input is (B, bins, channels)
    """

    def __init__(self, in_ch: int = 3, d_out: int = 128, dropout: float = 0.1, num_bins: int = 96, **kwargs):
        super().__init__(**kwargs)
        self.in_ch = in_ch
        self.d_out = d_out
        self.out_dim = d_out
        self.dropout_rate = dropout
        self.num_bins = num_bins

        # Layer 1: Detect small local structures
        self.conv1 = keras.Sequential([
            layers.Conv1D(16, kernel_size=5, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPool1D(pool_size=2),
        ])

        # Layer 2: Detect larger structures
        self.conv2 = keras.Sequential([
            layers.Conv1D(32, kernel_size=5, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPool1D(pool_size=2),
        ])

        # Layer 3: High-level shape recognition
        self.conv3 = keras.Sequential([
            layers.Conv1D(64, kernel_size=3, padding="same"),
            layers.ReLU(),
            layers.GlobalAveragePooling1D(),
        ])

        # Final projection
        self.fc = keras.Sequential([
            layers.Dense(d_out, activation="relu"),
            layers.Dropout(dropout),
        ])

    def call(self, x_profile, training=None):
        # x_profile: (B, in_ch, num_bins) in PyTorch format
        # Transpose to (B, num_bins, in_ch) for TensorFlow
        x = tf.transpose(x_profile, [0, 2, 1])
        h = self.conv1(x, training=training)
        h = self.conv2(h, training=training)
        h = self.conv3(h, training=training)
        z = self.fc(h, training=training)
        return z

    def get_config(self):
        config = super().get_config()
        config.update({
            "in_ch": self.in_ch,
            "d_out": self.d_out,
            "dropout": self.dropout_rate,
            "num_bins": self.num_bins,
        })
        return config


class SeqEncoder(layers.Layer):
    """
    Sequential VPIN/bucketed flow encoder.

    x_seq: (B, T, F_seq) padded
    seq_len: (B,)
    -> (B, D)
    """

    def __init__(self, f_in: int, d_out: int = 128, d_conv: int = 64, d_lstm: int = 128,
                 dropout: float = 0.1, bidir: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.f_in = f_in
        self.d_out = d_out
        self.out_dim = d_out
        self.d_conv = d_conv
        self.d_lstm = d_lstm
        self.dropout_rate = dropout
        self.bidir = bidir

        self.pre = keras.Sequential([
            layers.LayerNormalization(),
            layers.Dense(d_conv, activation="gelu"),
        ])

        self.conv = layers.Conv1D(d_conv, kernel_size=5, padding="same")

        if bidir:
            self.lstm = layers.Bidirectional(
                layers.LSTM(d_lstm, return_sequences=True)
            )
            lstm_dim = d_lstm * 2
        else:
            self.lstm = layers.LSTM(d_lstm, return_sequences=True)
            lstm_dim = d_lstm

        self.post = keras.Sequential([
            layers.Dense(d_out, activation="gelu"),
            layers.Dropout(dropout),
        ])

        self.pool = AttnPool(d_out)

    def call(self, inputs, training=None):
        x_seq, seq_len = inputs
        # x_seq: (B, T, F)
        B = tf.shape(x_seq)[0]
        T = tf.shape(x_seq)[1]

        h = self.pre(x_seq, training=training)  # (B, T, d_conv)
        h = self.conv(h, training=training)  # (B, T, d_conv)

        # Create mask for LSTM
        mask = tf.sequence_mask(seq_len, maxlen=T, dtype=tf.bool)

        out = self.lstm(h, mask=mask, training=training)
        out = self.post(out, training=training)  # (B, T, D)

        # Create mask for pooling (1=valid, 0=pad)
        pool_mask = tf.cast(mask, tf.int32)
        z = self.pool(out, mask=pool_mask, training=training)  # (B, D)
        return z

    def get_config(self):
        config = super().get_config()
        config.update({
            "f_in": self.f_in,
            "d_out": self.d_out,
            "d_conv": self.d_conv,
            "d_lstm": self.d_lstm,
            "dropout": self.dropout_rate,
            "bidir": self.bidir,
        })
        return config


class NumberBarsEncoder(layers.Layer):
    """
    CNN-Transformer hybrid to process NumbersBars.

    x_nb: (B, T, BINS, C) -> (B, d_model)
    """

    def __init__(self, c_in: int = 3, d_model: int = 128, conv_ch: int = 64,
                 n_tf: int = 2, n_heads: int = 4, dropout: float = 0.1, max_T: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.c_in = c_in
        self.d_model = d_model
        self.out_dim = d_model
        self.conv_ch = conv_ch
        self.n_tf = n_tf
        self.n_heads = n_heads
        self.dropout_rate = dropout
        self.max_T = max_T

        # Per-slice conv encoder
        self.proj = layers.Conv1D(conv_ch, kernel_size=1)
        self.conv = keras.Sequential([
            layers.Conv1D(conv_ch, kernel_size=5, padding="same", activation="gelu"),
            layers.Dropout(dropout),
            layers.Conv1D(conv_ch, kernel_size=5, padding="same", activation="gelu"),
            layers.Dropout(dropout),
        ])
        self.pool_bins = layers.GlobalAveragePooling1D()
        self.slice_to_model = keras.Sequential([
            layers.Dense(d_model, activation="gelu"),
            layers.Dropout(dropout),
        ])

        # Positional embedding
        self.pos_embedding = layers.Embedding(max_T, d_model)

        # Transformer encoder layers
        self.transformer_layers = [
            layers.MultiHeadAttention(
                num_heads=n_heads,
                key_dim=d_model // n_heads,
                dropout=dropout
            )
            for _ in range(n_tf)
        ]
        self.ff_layers = [
            keras.Sequential([
                layers.LayerNormalization(),
                layers.Dense(4 * d_model, activation="gelu"),
                layers.Dropout(dropout),
                layers.Dense(d_model),
            ])
            for _ in range(n_tf)
        ]
        self.ln_layers = [layers.LayerNormalization() for _ in range(n_tf)]
        self.final_norm = layers.LayerNormalization()

    def call(self, inputs, training=None):
        x_nb, nb_lengths = inputs if isinstance(inputs, (list, tuple)) else (inputs, None)
        # x_nb: (B, T, BINS, C)
        B = tf.shape(x_nb)[0]
        T = tf.shape(x_nb)[1]
        BINS = tf.shape(x_nb)[2]
        C = tf.shape(x_nb)[3]

        # Reshape to (B*T, BINS, C) for conv processing
        x = tf.reshape(x_nb, [B * T, BINS, C])

        x = self.proj(x, training=training)  # (B*T, BINS, conv_ch)
        x = self.conv(x, training=training)  # (B*T, BINS, conv_ch)
        x = self.pool_bins(x)  # (B*T, conv_ch)
        x = self.slice_to_model(x, training=training)  # (B*T, d_model)
        x = tf.reshape(x, [B, T, self.d_model])  # (B, T, d_model)

        # Create attention mask if needed
        attention_mask = None
        if nb_lengths is not None:
            # Create mask: True where PAD
            t_range = tf.range(T)[None, :]  # (1, T)
            padding_mask = t_range >= nb_lengths[:, None]  # (B, T)
            # Expand for attention: (B, 1, 1, T)
            attention_mask = tf.cast(padding_mask[:, None, None, :], tf.float32) * -1e9

        # Add positional embeddings
        pos = tf.range(T)[None, :]  # (1, T)
        pos = tf.broadcast_to(pos, [B, T])
        x = x + self.pos_embedding(pos)

        # Transformer layers
        for attn, ff, ln in zip(self.transformer_layers, self.ff_layers, self.ln_layers):
            # Pre-norm attention
            x_norm = ln(x)
            attn_out = attn(x_norm, x_norm, attention_mask=attention_mask, training=training)
            x = x + attn_out
            # FFN with residual
            x = x + ff(x, training=training)

        # Masked mean pool
        if nb_lengths is None:
            z = tf.reduce_mean(x, axis=1)
        else:
            t_range = tf.range(T)[None, :]
            mask = tf.cast(t_range < nb_lengths[:, None], tf.float32)  # (B, T)
            x_masked = x * mask[:, :, None]  # (B, T, d_model)
            z = tf.reduce_sum(x_masked, axis=1) / tf.maximum(tf.reduce_sum(mask, axis=1, keepdims=True), 1.0)

        return self.final_norm(z)

    def get_config(self):
        config = super().get_config()
        config.update({
            "c_in": self.c_in,
            "d_model": self.d_model,
            "conv_ch": self.conv_ch,
            "n_tf": self.n_tf,
            "n_heads": self.n_heads,
            "dropout": self.dropout_rate,
            "max_T": self.max_T,
        })
        return config


class SummaryEncoder(layers.Layer):
    """Summary: (B, F_sum) -> (B, D)"""

    def __init__(self, f_in: int, d_out: int = 128, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.f_in = f_in
        self.d_out = d_out
        self.out_dim = d_out
        self.dropout_rate = dropout

        self.net = keras.Sequential([
            layers.LayerNormalization(),
            layers.Dense(256, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(d_out, activation="gelu"),
        ])

    def call(self, x_sum, training=None):
        return self.net(x_sum, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "f_in": self.f_in,
            "d_out": self.d_out,
            "dropout": self.dropout_rate,
        })
        return config


class SummaryMLPEnc(layers.Layer):
    """Simple MLP encoder for summary features (matches DualBranchModel default)."""

    def __init__(self, f_in: int, d_hidden: int = 128, dropout: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.f_in = f_in
        self.d_hidden = d_hidden
        self.out_dim = d_hidden // 2
        self.dropout_rate = dropout

        self.net = keras.Sequential([
            layers.Dense(d_hidden),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(dropout),
            layers.Dense(self.out_dim, activation="relu"),
        ])

    def call(self, x, training=None):
        return self.net(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "f_in": self.f_in,
            "d_hidden": self.d_hidden,
            "dropout": self.dropout_rate,
        })
        return config


# ----------------------------
# ResNet Components
# ----------------------------


class SEBlock1D(layers.Layer):
    """Squeeze-and-Excitation Block for 1D."""

    def __init__(self, channel: int, reduction: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.channel = channel
        self.reduction = reduction
        self.pool = layers.GlobalAveragePooling1D()
        self.fc = keras.Sequential([
            layers.Dense(channel // reduction, activation="relu", use_bias=False),
            layers.Dense(channel, activation="sigmoid", use_bias=False),
        ])

    def call(self, x, training=None):
        b = tf.shape(x)[0]
        y = self.pool(x)  # (B, C)
        y = self.fc(y)  # (B, C)
        y = tf.reshape(y, [b, 1, self.channel])  # (B, 1, C)
        return x * y

    def get_config(self):
        config = super().get_config()
        config.update({"channel": self.channel, "reduction": self.reduction})
        return config


class ResBlock1D(layers.Layer):
    """1D Residual Block with optional SE attention."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, downsample: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.do_downsample = downsample

        # Always use "same" padding - TensorFlow handles this correctly with strides
        self.conv1 = layers.Conv1D(out_channels, kernel_size, strides=stride,
                                   padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(out_channels, kernel_size, padding="same", use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.se = SEBlock1D(out_channels)

        # Downsample shortcut when stride > 1 or channels change
        if stride != 1 or in_channels != out_channels:
            self.downsample = keras.Sequential([
                layers.Conv1D(out_channels, kernel_size=1, strides=stride,
                              padding="same", use_bias=False),
                layers.BatchNormalization(),
            ])
        else:
            self.downsample = None

    def call(self, x, training=None):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = tf.nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.se(out, training=training)

        if self.downsample is not None:
            residual = self.downsample(x, training=training)

        out = out + residual
        out = tf.nn.relu(out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "downsample": self.do_downsample,
        })
        return config


class MarketProfileResNet(layers.Layer):
    """
    Enhanced Market Profile Encoder.
    ResNet-1D + SE Attention for profile data.

    Input: (B, C, Bins) in PyTorch format -> transposed to (B, Bins, C)
    Output: (B, d_model)
    """

    def __init__(self, in_channels: int = 3, d_model: int = 128, layer_blocks=(2, 2, 2), **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.d_model = d_model
        self.layer_blocks = layer_blocks
        self.out_dim = d_model

        # Stem
        self.conv1 = layers.Conv1D(32, kernel_size=7, strides=2, padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.maxpool = layers.MaxPool1D(pool_size=3, strides=2, padding="same")

        # ResNet layers
        self.res_layers = []
        in_ch = 32
        for i, (out_ch, blocks) in enumerate(zip([32, 64, 128], layer_blocks)):
            for j in range(blocks):
                stride = 2 if j == 0 and i > 0 else 1
                downsample = (j == 0 and (stride != 1 or in_ch != out_ch))
                self.res_layers.append(ResBlock1D(in_ch, out_ch, stride=stride, downsample=downsample))
                in_ch = out_ch

        # Output head
        self.avgpool = layers.GlobalAveragePooling1D()
        self.fc = layers.Dense(d_model)
        self.norm = layers.LayerNormalization()

    def call(self, x, training=None):
        # x: (B, C, Bins) in PyTorch format
        # Transpose to (B, Bins, C) for TensorFlow
        x = tf.transpose(x, [0, 2, 1])

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.maxpool(x)

        for res_layer in self.res_layers:
            x = res_layer(x, training=training)

        x = self.avgpool(x)
        x = self.fc(x)
        return self.norm(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "in_channels": self.in_channels,
            "d_model": self.d_model,
            "layer_blocks": self.layer_blocks,
        })
        return config


class BasicBlock2D(layers.Layer):
    """Standard ResNet Basic Block for 2D (Time x Bins)."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

        self.conv1 = layers.Conv2D(planes, kernel_size=3, strides=stride, padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(planes, kernel_size=3, padding="same", use_bias=False)
        self.bn2 = layers.BatchNormalization()

        if stride != 1 or in_planes != planes:
            self.shortcut = keras.Sequential([
                layers.Conv2D(planes, kernel_size=1, strides=stride,
                              padding="same", use_bias=False),
                layers.BatchNormalization(),
            ])
        else:
            self.shortcut = None

    def call(self, x, training=None):
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = self.bn2(self.conv2(out), training=training)
        shortcut = self.shortcut(x, training=training) if self.shortcut else x
        out = out + shortcut
        return tf.nn.relu(out)

    def get_config(self):
        config = super().get_config()
        config.update({
            "in_planes": self.in_planes,
            "planes": self.planes,
            "stride": self.stride,
        })
        return config


class RasterResNet(layers.Layer):
    """
    "3D" ResNet implementation for Rasterized VPIN data.

    Input: (B, T, C, Bins) in PyTorch format -> transposed to (B, T, Bins, C)
    Output: (B, d_model)
    """

    def __init__(self, in_ch: int = 4, d_model: int = 128, layer_blocks=(2, 2, 2),
                 base_filters: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.in_ch = in_ch
        self.d_model = d_model
        self.out_dim = d_model
        self.layer_blocks = layer_blocks
        self.base_filters = base_filters

        # Stem
        self.conv1 = layers.Conv2D(base_filters, kernel_size=3, padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization()

        # ResNet layers
        self.res_layers = []
        in_planes = base_filters
        filter_sizes = [base_filters, base_filters * 2, base_filters * 4]
        strides = [1, 2, 2]

        for i, (filters, num_blocks, stride) in enumerate(zip(filter_sizes, layer_blocks, strides)):
            for j in range(num_blocks):
                s = stride if j == 0 else 1
                self.res_layers.append(BasicBlock2D(in_planes, filters, stride=s))
                in_planes = filters

        # Aggregation
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(d_model)
        self.norm = layers.LayerNormalization()

    def call(self, x, lengths=None, training=None):
        # x: (B, T, C, Bins) in PyTorch format
        # Transpose to (B, T, Bins, C) for TensorFlow Conv2D (channels last)
        x = tf.transpose(x, [0, 1, 3, 2])

        x = tf.nn.relu(self.bn1(self.conv1(x), training=training))

        for res_layer in self.res_layers:
            x = res_layer(x, training=training)

        x = self.avgpool(x)
        x = self.fc(x)
        return self.norm(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "in_ch": self.in_ch,
            "d_model": self.d_model,
            "layer_blocks": self.layer_blocks,
            "base_filters": self.base_filters,
        })
        return config


# ----------------------------
# Spatial-Temporal Encoder
# ----------------------------


class PositionalEncoding(layers.Layer):
    """Sinusoidal positional encoding for transformer sequences."""

    def __init__(self, d_model: int, max_len: int = 5000, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len

        # Build positional encoding
        position = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * (-math.log(10000.0) / d_model))
        pe = tf.zeros((max_len, d_model))

        indices_even = tf.range(0, d_model, 2)
        indices_odd = tf.range(1, d_model, 2)

        sin_vals = tf.sin(position * div_term)
        cos_vals = tf.cos(position * div_term)

        # Use scatter_nd to build the encoding
        pe_even = sin_vals
        pe_odd = cos_vals

        self.pe = tf.Variable(
            tf.concat([
                tf.reshape(pe_even, [max_len, -1, 1]),
                tf.reshape(pe_odd, [max_len, -1, 1])
            ], axis=-1),
            trainable=False
        )
        # Flatten back properly
        self.pe = tf.Variable(
            tf.reshape(self.pe, [max_len, d_model]),
            trainable=False
        )

    def call(self, x, training=None):
        # x: (B, Seq, D)
        seq_len = tf.shape(x)[1]
        return x + self.pe[:seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "max_len": self.max_len})
        return config


class SpatialTemporalEncoder(layers.Layer):
    """
    CNN-Transformer hybrid for rasterized VPIN sequences.

    Input: (B, T, C, Bins) in PyTorch format -> processed internally
    Output: (B, d_model)
    """

    def __init__(
        self,
        num_bars: int = 4,
        in_ch: int = 4,
        num_bins: int = 64,
        d_model: int = 128,
        nhead: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.out_dim = d_model
        self.num_bars = num_bars
        self.num_bins = num_bins
        self.d_model = d_model

        # Spatial encoder (CNN) - processes each time bar independently
        self.cnn = keras.Sequential([
            layers.Conv1D(32, kernel_size=3, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("gelu"),
            layers.MaxPool1D(2),

            layers.Conv1D(64, kernel_size=3, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("gelu"),
            layers.MaxPool1D(2),

            layers.Conv1D(d_model, kernel_size=3, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("gelu"),
        ])

        # Adapter to create bar embeddings
        cnn_out_size = d_model * (num_bins // 4)
        self.adapter = keras.Sequential([
            layers.Flatten(),
            layers.Dense(d_model),
            layers.LayerNormalization(),
            layers.Activation("gelu"),
        ])

        # Temporal encoder (Transformer)
        self.pos_encoder = PositionalEncoding(d_model, max_len=num_bars)

        self.transformer_layers = []
        for _ in range(n_layers):
            self.transformer_layers.append({
                'attn': layers.MultiHeadAttention(
                    num_heads=nhead,
                    key_dim=d_model // nhead,
                    dropout=dropout
                ),
                'ffn': keras.Sequential([
                    layers.Dense(4 * d_model, activation="gelu"),
                    layers.Dropout(dropout),
                    layers.Dense(d_model),
                ]),
                'ln1': layers.LayerNormalization(),
                'ln2': layers.LayerNormalization(),
            })

        # Attention pooling
        self.attn_pool = layers.Dense(1)
        self.norm = layers.LayerNormalization()

    def call(self, x, lengths=None, training=None):
        # x: (B, T, C, Bins) in PyTorch format
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        # Transpose to (B, T, Bins, C) for TensorFlow
        x = tf.transpose(x, [0, 1, 3, 2])

        # Reshape to (B*T, Bins, C) for CNN
        x_flat = tf.reshape(x, [B * T, self.num_bins, -1])

        # CNN feature extraction
        cnn_feat = self.cnn(x_flat, training=training)  # (B*T, bins/4, d_model)

        # Create bar embeddings
        bar_embeds = self.adapter(cnn_feat, training=training)  # (B*T, d_model)

        # Unfold time
        x_seq = tf.reshape(bar_embeds, [B, T, self.d_model])  # (B, T, d_model)

        # Add positional encoding
        x_seq = self.pos_encoder(x_seq, training=training)

        # Attention mask
        attention_mask = None
        if lengths is not None:
            t_range = tf.range(T)[None, :]
            padding_mask = t_range >= lengths[:, None]
            attention_mask = tf.cast(padding_mask[:, None, None, :], tf.float32) * -1e9

        # Transformer layers
        memory = x_seq
        for layer in self.transformer_layers:
            # Pre-norm attention
            x_norm = layer['ln1'](memory)
            attn_out = layer['attn'](x_norm, x_norm, attention_mask=attention_mask, training=training)
            memory = memory + attn_out
            # FFN
            x_norm = layer['ln2'](memory)
            memory = memory + layer['ffn'](x_norm, training=training)

        # Attention pooling
        scores = self.attn_pool(memory)  # (B, T, 1)
        if lengths is not None:
            t_range = tf.range(T)[None, :]
            mask = tf.cast(t_range >= lengths[:, None], tf.float32) * -1e9
            scores = scores + mask[:, :, None]

        weights = tf.nn.softmax(scores, axis=1)
        context = tf.reduce_sum(memory * weights, axis=1)  # (B, d_model)

        return self.norm(context)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_bars": self.num_bars,
            "in_ch": 4,
            "num_bins": self.num_bins,
            "d_model": self.d_model,
        })
        return config


class MarketProfileCNN(layers.Layer):
    """
    A lightweight 1D CNN for Market Profile (histogram) data.

    Input: (B, C, Bins) in PyTorch format -> transposed internally
    Output: (B, out_dim)
    """

    def __init__(self, in_channels: int, out_dim: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_dim = out_dim

        self.net = keras.Sequential([
            # Block 1: Capture local shape
            layers.Conv1D(16, kernel_size=5, padding="same"),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),
            layers.MaxPool1D(2),

            # Block 2: Capture structure
            layers.Conv1D(32, kernel_size=5, padding="same"),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),
            layers.MaxPool1D(2),

            # Block 3: Global abstraction
            layers.Conv1D(64, kernel_size=3, padding="same"),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),
            layers.GlobalAveragePooling1D(),
        ])

        self.fc = layers.Dense(out_dim)

    def call(self, x, training=None):
        # x: (B, C, Bins) in PyTorch format
        # Transpose to (B, Bins, C)
        x = tf.transpose(x, [0, 2, 1])
        x = self.net(x, training=training)
        return self.fc(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "in_channels": self.in_channels,
            "out_dim": self.out_dim,
        })
        return config


class IntradayRNN(layers.Layer):
    """
    Encoder for the Sequential branch (Intraday VPIN time-series).

    Uses GRU for processing variable-length sequences.
    """

    def __init__(self, input_dim: int, d_model: int = 128, num_layers: int = 1,
                 dropout: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.d_model = d_model
        self.out_dim = d_model
        self.num_layers = num_layers
        self.dropout_rate = dropout

        if num_layers == 1:
            self.rnn = layers.GRU(d_model, return_sequences=False, return_state=True)
        else:
            # Stack GRUs for multiple layers
            self.rnn_layers = []
            for i in range(num_layers - 1):
                self.rnn_layers.append(
                    layers.GRU(d_model, return_sequences=True, dropout=dropout)
                )
            self.rnn_layers.append(
                layers.GRU(d_model, return_sequences=False, return_state=True)
            )
            self.rnn = None

        self.norm = layers.LayerNormalization()

    def call(self, x, lengths=None, training=None):
        # x: (B, SeqLen, Features)
        mask = None
        if lengths is not None:
            T = tf.shape(x)[1]
            mask = tf.sequence_mask(lengths, maxlen=T, dtype=tf.bool)

        if self.rnn is not None:
            # Single layer
            output, h_n = self.rnn(x, mask=mask, training=training)
            embedding = h_n
        else:
            # Multiple layers
            h = x
            for i, rnn_layer in enumerate(self.rnn_layers[:-1]):
                h = rnn_layer(h, mask=mask, training=training)
            output, h_n = self.rnn_layers[-1](h, mask=mask, training=training)
            embedding = h_n

        return self.norm(embedding)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
        })
        return config
