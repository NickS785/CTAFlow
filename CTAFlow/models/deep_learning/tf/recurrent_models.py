"""
TensorFlow/Keras implementations of recurrent multi-modal models.

Migrated from CTAFlow.models.deep_learning.multi_branch (PyTorch).

Models:
    - RecurrentDualModal: Summary + Profile + Raster with Window LSTM
    - RecurrentWSPR: Windowed Summary, Profile, Recent Raster/Sequential model
    - RecurrentTriModal: Full tri-modal with Summary + Profile + Raster + Sequential
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .encoders import (
    MarketProfileResNet,
    RasterResNet,
    SpatialFuse,
    IntradayRNN,
)


class RecurrentDualModal(keras.Model):
    """
    State-of-the-art Sequence Model for Market Data (TensorFlow/Keras).

    Components:
    1. Macro: Summary MLP
    2. Structure (Static): MarketProfileResNet (1D ResNet + SE)
    3. Flow (Dynamic): RasterResNet (Pseudo-3D ResNet)
    4. Sequence: Window LSTM

    Input shapes (PyTorch format, transposed internally):
    - summary_days: (B, W, f_sum)
    - profile_days: (B, W, f_prof, bins)
    - raster_days: (B, W, T_bars, f_rast, bins)

    Output: (B, num_classes) or (B, 1)
    """

    def __init__(
        self,
        f_sum: int,
        f_profile: int = 3,
        f_raster: int = 4,
        num_bins: int = 64,
        d_model: int = 128,
        lstm_hidden: int = 128,
        spatial_fuse_mode: str = "gated",
        task: str = "classification",
        num_classes: int = 3,
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.f_sum = f_sum
        self.f_profile = f_profile
        self.f_raster = f_raster
        self.num_bins = num_bins
        self.d_model = d_model
        self.lstm_hidden = lstm_hidden
        self.task = task
        self.num_classes = num_classes
        self.dropout_rate = dropout

        # --- 1. Summary Encoder ---
        self.summary_net = keras.Sequential([
            layers.Dense(d_model // 2),
            layers.LayerNormalization(),
            layers.Activation("gelu"),
            layers.Dropout(dropout),
            layers.Dense(d_model // 2, activation="gelu"),
        ], name="summary_encoder")

        # --- 2. Enhanced Spatial Encoders ---

        # A. Profile (Static Structure)
        self.profile_net = MarketProfileResNet(
            in_channels=f_profile,
            d_model=d_model // 2,
            layer_blocks=[2, 2, 2]
        )

        # B. Raster (Dynamic Flow)
        self.raster_net = RasterResNet(
            in_ch=f_raster,
            d_model=d_model // 2,
            layer_blocks=[2, 2, 2],
            base_filters=32
        )

        # C. Fusion
        self.spatial_fuse = SpatialFuse(
            d_spatial=d_model // 2,
            mode=spatial_fuse_mode
        )

        # --- 3. Day Fusion & LSTM ---
        self.day_fuse = keras.Sequential([
            layers.Dense(d_model),
            layers.LayerNormalization(),
            layers.Activation("gelu"),
            layers.Dropout(dropout),
        ], name="day_fusion")

        # Window LSTM
        self.window_lstm = layers.LSTM(lstm_hidden, return_sequences=False, return_state=True)

        # --- 4. Prediction Head ---
        out_units = num_classes if task == "classification" else 1
        self.head = keras.Sequential([
            layers.Dense(64, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(out_units),
        ], name="prediction_head")

    def call(self, inputs, training=None, return_probs=False):
        """
        Forward pass.

        Parameters
        ----------
        inputs : tuple
            (summary_days, profile_days, raster_days) where:
            - summary_days: (B, W, f_sum)
            - profile_days: (B, W, f_prof, bins)
            - raster_days: (B, W, T_bars, f_rast, bins)
        training : bool, optional
            Training mode flag
        return_probs : bool, default False
            If True and task is classification, return softmax probabilities

        Returns
        -------
        tf.Tensor
            Logits or probabilities, shape (B, num_classes) or (B, 1)
        """
        summary_days, profile_days, raster_days = inputs

        B = tf.shape(profile_days)[0]
        W = tf.shape(profile_days)[1]
        BW = B * W

        # 1. Flatten Time for shared encoders
        flat_sum = tf.reshape(summary_days, [BW, -1])
        flat_prof = tf.reshape(profile_days, [BW, tf.shape(profile_days)[2], -1])
        flat_rast = tf.reshape(raster_days, [BW, tf.shape(raster_days)[2],
                                              tf.shape(raster_days)[3], -1])

        # 2. Run Encoders
        z_sum = self.summary_net(flat_sum, training=training)  # (BW, d/2)
        z_prof = self.profile_net(flat_prof, training=training)  # (BW, d/2)
        z_rast = self.raster_net(flat_rast, training=training)  # (BW, d/2)

        # 3. Spatial Fusion (Static + Dynamic)
        z_spatial = self.spatial_fuse(z_prof, z_rast, training=training)  # (BW, d/2)

        # 4. Day Fusion (Macro + Spatial)
        z_day_cat = tf.concat([z_sum, z_spatial], axis=1)  # (BW, d)
        z_day = self.day_fuse(z_day_cat, training=training)  # (BW, d)

        # 5. Window LSTM
        z_seq = tf.reshape(z_day, [B, W, -1])  # (B, W, d)
        lstm_out, h_n, _ = self.window_lstm(z_seq, training=training)
        z_final = h_n  # (B, lstm_hidden)

        # 6. Prediction
        logits = self.head(z_final, training=training)

        if self.task == "classification" and return_probs:
            return tf.nn.softmax(logits, axis=-1)

        return logits

    def get_config(self):
        config = super().get_config()
        config.update({
            "f_sum": self.f_sum,
            "f_profile": self.f_profile,
            "f_raster": self.f_raster,
            "num_bins": self.num_bins,
            "d_model": self.d_model,
            "lstm_hidden": self.lstm_hidden,
            "task": self.task,
            "num_classes": self.num_classes,
            "dropout": self.dropout_rate,
        })
        return config


class RecurrentWSPR(keras.Model):
    """
    Recurrent Windowed Summary, Profile, and Recent Raster/Sequential/Fused model.

    This model processes inputs through multiple parallel paths:
    - Path 1 (Windowed Summary): LSTM over a window of summary embeddings.
    - Path 2 (Windowed Profile): LSTM over a window of profile embeddings.
    - Path 3 (Recent Raster): Encoder processes only the most recent raster data.
    - Path 4 (Recent Sequential): Encoder processes only the most recent sequential data.
    - Path 5 (Recent Spatial Fusion): Spatially fuses the most recent profile and raster.

    Input shapes (PyTorch format):
    - summary_days: (B, W, f_sum)
    - profile_days: (B, W, f_profile, bins)
    - raster_recent: (B, T_bars, f_raster, bins) - most recent day only
    - seq_recent: (B, SeqLen, f_seq) - most recent day only
    - seq_lens_recent: (B,) - sequence lengths

    Output: (B, num_classes) or (B, 1)
    """

    def __init__(
        self,
        f_sum: int,
        f_profile: int,
        f_raster: int,
        f_seq: int,
        d_model: int = 128,
        sum_lstm_hidden: int = 64,
        prof_lstm_hidden: int = 128,
        task: str = "classification",
        num_classes: int = 3,
        dropout: float = 0.3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.f_sum = f_sum
        self.f_profile = f_profile
        self.f_raster = f_raster
        self.f_seq = f_seq
        self.d_model = d_model
        self.task = task
        self.num_classes = num_classes
        self.dropout_rate = dropout

        # --- Branch Encoders ---
        self.summary_net = keras.Sequential([
            layers.Dense(d_model),
            layers.LayerNormalization(),
            layers.Activation("gelu"),
        ], name="summary_encoder")

        self.profile_net = MarketProfileResNet(
            in_channels=f_profile,
            d_model=d_model
        )

        self.raster_net = RasterResNet(
            in_ch=f_raster,
            d_model=d_model
        )

        self.seq_net = IntradayRNN(
            input_dim=f_seq,
            d_model=d_model,
            num_layers=1
        )

        self.spatial_fuse = SpatialFuse(
            d_spatial=d_model,
            mode="gated"
        )

        # --- Path 1 & 2: Windowed LSTMs ---
        self.summary_lstm = layers.LSTM(sum_lstm_hidden, return_sequences=False, return_state=True)
        self.profile_lstm = layers.LSTM(prof_lstm_hidden, return_sequences=False, return_state=True)

        # --- Final Fusion Head ---
        final_fusion_dim = (
            sum_lstm_hidden      # Path 1
            + prof_lstm_hidden   # Path 2
            + d_model            # Path 3 (Recent Raster)
            + d_model            # Path 4 (Recent Seq)
            + d_model            # Path 5 (Recent Spatial Fuse)
        )

        out_units = num_classes if task == "classification" else 1
        self.head = keras.Sequential([
            layers.Dense(256),
            layers.LayerNormalization(),
            layers.Activation("gelu"),
            layers.Dropout(dropout),
            layers.Dense(128, activation="gelu"),
            layers.Dense(out_units),
        ], name="prediction_head")

    def call(self, inputs, training=None, return_probs=False):
        """
        Forward pass.

        Parameters
        ----------
        inputs : tuple
            (summary_days, profile_days, raster_recent, seq_recent, seq_lens_recent)
        training : bool, optional
            Training mode flag
        return_probs : bool, default False
            If True and task is classification, return softmax probabilities

        Returns
        -------
        tf.Tensor
            Logits or probabilities
        """
        summary_days, profile_days, raster_recent, seq_recent, seq_lens_recent = inputs

        B = tf.shape(summary_days)[0]
        W = tf.shape(summary_days)[1]
        BW = B * W

        # --- Encode all days for windowed paths ---
        flat_sum = tf.reshape(summary_days, [BW, -1])
        z_sum_all = self.summary_net(flat_sum, training=training)  # (BW, d)

        flat_prof = tf.reshape(profile_days, [BW, tf.shape(profile_days)[2], -1])
        z_prof_all = self.profile_net(flat_prof, training=training)  # (BW, d)

        # --- Path 1: Windowed Summary LSTM ---
        z_sum_seq = tf.reshape(z_sum_all, [B, W, -1])
        _, h_n_sum, _ = self.summary_lstm(z_sum_seq, training=training)
        z_summary_temporal = h_n_sum  # (B, sum_lstm_hidden)

        # --- Path 2: Windowed Profile LSTM ---
        z_prof_seq = tf.reshape(z_prof_all, [B, W, -1])
        _, h_n_prof, _ = self.profile_lstm(z_prof_seq, training=training)
        z_profile_temporal = h_n_prof  # (B, prof_lstm_hidden)

        # --- Path 3: Recent Raster ---
        z_raster_recent = self.raster_net(raster_recent, training=training)  # (B, d)

        # --- Path 4: Recent Sequential ---
        z_seq_recent = self.seq_net(seq_recent, lengths=seq_lens_recent, training=training)  # (B, d)

        # --- Path 5: Recent Spatial Fusion ---
        # Get profile embedding for most recent day
        z_prof_recent = z_prof_seq[:, -1, :]  # (B, d)
        z_spatial_fused_recent = self.spatial_fuse(z_prof_recent, z_raster_recent, training=training)  # (B, d)

        # --- Final Concatenation ---
        z_final_cat = tf.concat([
            z_summary_temporal,
            z_profile_temporal,
            z_raster_recent,
            z_seq_recent,
            z_spatial_fused_recent
        ], axis=1)

        # --- Prediction Head ---
        logits = self.head(z_final_cat, training=training)

        if self.task == "classification" and return_probs:
            return tf.nn.softmax(logits, axis=-1)

        return logits

    def get_config(self):
        config = super().get_config()
        config.update({
            "f_sum": self.f_sum,
            "f_profile": self.f_profile,
            "f_raster": self.f_raster,
            "f_seq": self.f_seq,
            "d_model": self.d_model,
            "task": self.task,
            "num_classes": self.num_classes,
            "dropout": self.dropout_rate,
        })
        return config


class RecurrentTriModal(keras.Model):
    """
    State-of-the-Art Recurrent Tri-Modal Model (TensorFlow/Keras).

    Processing Pipeline:
    1. Flatten Window Dimension (Batch * Days).
    2. Encode Daily Modalities independently:
        - Summary -> MLP
        - Profile -> MarketProfileResNet
        - Raster -> RasterResNet
        - Seq -> IntradayRNN
    3. Spatial Fusion: Merge Profile + Raster.
    4. Day Fusion: Merge Summary + Spatial + Seq.
    5. Window Modeling: Pass sequence of Day Embeddings to Window-LSTM.
    6. Prediction Head.

    Input shapes (PyTorch format):
    - summary_days: (B, W, f_sum)
    - seq_days: (B, W, SeqLen, f_seq)
    - seq_lens: (B, W) - sequence lengths per day
    - profile_days: (B, W, f_prof, bins)
    - raster_days: (B, W, T_bars, f_rast, bins)

    Output: (B, num_classes) or (B, 1)
    """

    def __init__(
        self,
        f_sum: int,
        f_seq: int,
        f_profile: int = 3,
        f_raster: int = 4,
        d_model: int = 128,
        lstm_hidden: int = 128,
        head_dropout: float = 0.2,
        fusion_dropout: float = 0.3,
        task: str = "classification",
        num_classes: int = 3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.f_sum = f_sum
        self.f_seq = f_seq
        self.f_profile = f_profile
        self.f_raster = f_raster
        self.d_model = d_model
        self.lstm_hidden = lstm_hidden
        self.task = task
        self.num_classes = num_classes
        self.head_dropout = head_dropout
        self.fusion_dropout = fusion_dropout

        # --- Branch 1: Summary (Macro) ---
        self.summary_net = keras.Sequential([
            layers.Dense(d_model),
            layers.LayerNormalization(),
            layers.Activation("gelu"),
            layers.Dropout(fusion_dropout),
            layers.Dense(d_model),
        ], name="summary_encoder")

        # --- Branch 2: Spatial System (Structure) ---
        self.profile_net = MarketProfileResNet(
            in_channels=f_profile,
            d_model=d_model
        )

        self.raster_net = RasterResNet(
            in_ch=f_raster,
            d_model=d_model
        )

        self.spatial_fuse = SpatialFuse(
            d_spatial=d_model,
            mode="gated"
        )

        # --- Branch 3: Sequential (Intraday Flow) ---
        self.seq_net = IntradayRNN(
            input_dim=f_seq,
            d_model=d_model,
            num_layers=1
        )

        # --- Day Fusion ---
        # Combines Summary (d) + SpatialFused (d) + Sequential (d) = 3*d
        self.day_fuse = keras.Sequential([
            layers.Dense(d_model),
            layers.LayerNormalization(),
            layers.Activation("gelu"),
            layers.Dropout(fusion_dropout),
        ], name="day_fusion")

        # --- Window Sequence Modeling ---
        self.window_lstm = layers.LSTM(lstm_hidden, return_sequences=False, return_state=True)

        # --- Prediction Head ---
        out_units = num_classes if task == "classification" else 1
        self.head = keras.Sequential([
            layers.Dense(64, activation="gelu"),
            layers.Dropout(head_dropout),
            layers.Dense(out_units),
        ], name="prediction_head")

    def call(self, inputs, training=None, return_probs=False):
        """
        Forward pass handling 3 modalities over a window of days.

        Parameters
        ----------
        inputs : tuple
            (summary_days, seq_days, seq_lens, profile_days, raster_days)
        training : bool, optional
            Training mode flag
        return_probs : bool, default False
            If True and task is classification, return softmax probabilities

        Returns
        -------
        tf.Tensor
            Logits or probabilities
        """
        summary_days, seq_days, seq_lens, profile_days, raster_days = inputs

        B = tf.shape(summary_days)[0]
        W = tf.shape(summary_days)[1]
        BW = B * W

        # 1. Flatten Batch & Window dimensions
        flat_sum = tf.reshape(summary_days, [BW, -1])
        flat_prof = tf.reshape(profile_days, [BW, tf.shape(profile_days)[2], -1])

        # Raster: (B, W, T, C, Bins) -> (BW, T, C, Bins)
        flat_rast = tf.reshape(raster_days, [BW, tf.shape(raster_days)[2],
                                              tf.shape(raster_days)[3], -1])

        # Seq: (B, W, SeqLen, F) -> (BW, SeqLen, F)
        flat_seq = tf.reshape(seq_days, [BW, tf.shape(seq_days)[2], -1])

        # Handle lengths flattening
        flat_lens = None
        if seq_lens is not None:
            flat_lens = tf.reshape(seq_lens, [BW])

        # 2. Encode Branches
        z_sum = self.summary_net(flat_sum, training=training)  # (BW, d)
        z_prof = self.profile_net(flat_prof, training=training)  # (BW, d)
        z_rast = self.raster_net(flat_rast, training=training)  # (BW, d)
        z_seq = self.seq_net(flat_seq, lengths=flat_lens, training=training)  # (BW, d)

        # 3. Spatial Fusion
        z_spatial = self.spatial_fuse(z_prof, z_rast, training=training)  # (BW, d)

        # 4. Global Day Fusion
        # Concatenate: Summary | Spatial | Sequential
        z_day_cat = tf.concat([z_sum, z_spatial, z_seq], axis=1)  # (BW, 3d)
        z_day = self.day_fuse(z_day_cat, training=training)  # (BW, d)

        # 5. Window LSTM
        # Unflatten: (BW, d) -> (B, W, d)
        z_window_seq = tf.reshape(z_day, [B, W, -1])

        # LSTM over the window of days
        _, h_n, _ = self.window_lstm(z_window_seq, training=training)
        z_final = h_n  # (B, lstm_hidden)

        # 6. Prediction
        logits = self.head(z_final, training=training)

        if self.task == "classification" and return_probs:
            return tf.nn.softmax(logits, axis=-1)

        return logits

    def get_config(self):
        config = super().get_config()
        config.update({
            "f_sum": self.f_sum,
            "f_seq": self.f_seq,
            "f_profile": self.f_profile,
            "f_raster": self.f_raster,
            "d_model": self.d_model,
            "lstm_hidden": self.lstm_hidden,
            "task": self.task,
            "num_classes": self.num_classes,
            "head_dropout": self.head_dropout,
            "fusion_dropout": self.fusion_dropout,
        })
        return config


# Utility function to create model with functional API for more flexibility
def create_recurrent_dual_modal_functional(
    f_sum: int,
    window_size: int,
    f_profile: int = 3,
    profile_bins: int = 96,
    f_raster: int = 4,
    raster_bars: int = 4,
    raster_bins: int = 64,
    d_model: int = 128,
    lstm_hidden: int = 128,
    task: str = "classification",
    num_classes: int = 3,
    dropout: float = 0.2,
):
    """
    Create RecurrentDualModal using Keras Functional API.

    This allows for more explicit input shape specification and easier
    model inspection/visualization.

    Returns
    -------
    keras.Model
        Compiled model with explicit input layers
    """
    # Define inputs
    summary_input = layers.Input(shape=(window_size, f_sum), name="summary_days")
    profile_input = layers.Input(shape=(window_size, f_profile, profile_bins), name="profile_days")
    raster_input = layers.Input(shape=(window_size, raster_bars, f_raster, raster_bins), name="raster_days")

    # Create model
    model = RecurrentDualModal(
        f_sum=f_sum,
        f_profile=f_profile,
        f_raster=f_raster,
        num_bins=raster_bins,
        d_model=d_model,
        lstm_hidden=lstm_hidden,
        task=task,
        num_classes=num_classes,
        dropout=dropout,
    )

    # Build model
    outputs = model([summary_input, profile_input, raster_input])

    return keras.Model(
        inputs=[summary_input, profile_input, raster_input],
        outputs=outputs,
        name="RecurrentDualModal"
    )


def create_recurrent_tri_modal_functional(
    f_sum: int,
    f_seq: int,
    window_size: int,
    seq_len: int,
    f_profile: int = 3,
    profile_bins: int = 96,
    f_raster: int = 4,
    raster_bars: int = 4,
    raster_bins: int = 64,
    d_model: int = 128,
    lstm_hidden: int = 128,
    task: str = "classification",
    num_classes: int = 3,
    dropout: float = 0.2,
):
    """
    Create RecurrentTriModal using Keras Functional API.

    Returns
    -------
    keras.Model
        Compiled model with explicit input layers
    """
    # Define inputs
    summary_input = layers.Input(shape=(window_size, f_sum), name="summary_days")
    seq_input = layers.Input(shape=(window_size, seq_len, f_seq), name="seq_days")
    seq_lens_input = layers.Input(shape=(window_size,), name="seq_lens")
    profile_input = layers.Input(shape=(window_size, f_profile, profile_bins), name="profile_days")
    raster_input = layers.Input(shape=(window_size, raster_bars, f_raster, raster_bins), name="raster_days")

    # Create model
    model = RecurrentTriModal(
        f_sum=f_sum,
        f_seq=f_seq,
        f_profile=f_profile,
        f_raster=f_raster,
        d_model=d_model,
        lstm_hidden=lstm_hidden,
        task=task,
        num_classes=num_classes,
        fusion_dropout=dropout,
        head_dropout=dropout,
    )

    # Build model
    outputs = model([summary_input, seq_input, seq_lens_input, profile_input, raster_input])

    return keras.Model(
        inputs=[summary_input, seq_input, seq_lens_input, profile_input, raster_input],
        outputs=outputs,
        name="RecurrentTriModal"
    )
