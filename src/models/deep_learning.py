"""
Phase 4 (Layer 2) — Deep Learning Models.

  1. LSTM Autoencoder  – anomaly detection via reconstruction error
  2. LSTM Predictor    – time-series price forecasting; divergence = bubble signal
  3. Feedforward NN    – binary bubble classifier on macro features

All models are built with TensorFlow/Keras. The module degrades gracefully
if TensorFlow is not installed, returning None results.
"""

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Optional TensorFlow import
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks

    tf.get_logger().setLevel("ERROR")
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not installed — deep learning models will be skipped.")


# ============================================================
# Utilities
# ============================================================

def _make_sequences(data: np.ndarray, seq_len: int) -> np.ndarray:
    """Create sliding window sequences of shape (N, seq_len, features)."""
    seqs = []
    for i in range(len(data) - seq_len + 1):
        seqs.append(data[i : i + seq_len])
    return np.array(seqs)


def _early_stopping(patience: int = 15) -> "callbacks.EarlyStopping":
    return callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )


# ============================================================
# 1.  LSTM Autoencoder  (anomaly detection)
# ============================================================

@dataclass
class LSTMAutoencoderResult:
    reconstruction_error: pd.Series   # MSE per time step (aligned to input index)
    anomaly_flag: pd.Series            # True where error > threshold
    threshold: float
    model: object                      # keras model or None
    scaler: object


def build_lstm_autoencoder(n_features: int, seq_len: int, latent_dim: int = 16):
    """
    Encoder: LSTM (64) → LSTM (latent_dim)
    Decoder: RepeatVector → LSTM (64) → TimeDistributed Dense
    """
    if not TF_AVAILABLE:
        return None
    inputs = keras.Input(shape=(seq_len, n_features))
    encoded = layers.LSTM(64, return_sequences=True)(inputs)
    encoded = layers.LSTM(latent_dim, return_sequences=False)(encoded)
    decoded = layers.RepeatVector(seq_len)(encoded)
    decoded = layers.LSTM(64, return_sequences=True)(decoded)
    outputs = layers.TimeDistributed(layers.Dense(n_features))(decoded)
    model = keras.Model(inputs, outputs, name="lstm_autoencoder")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def train_lstm_autoencoder(
    feature_df: pd.DataFrame,
    seq_len: int = 12,
    epochs: int = 150,
    batch_size: int = 16,
    latent_dim: int = 16,
    save_path: Optional[Path] = None,
) -> LSTMAutoencoderResult:
    """
    Train LSTM Autoencoder on *normal* (non-bubble) windows.
    High reconstruction error on held-out data → anomalous (bubble) period.
    """
    dummy_result = LSTMAutoencoderResult(
        reconstruction_error=pd.Series(dtype=float),
        anomaly_flag=pd.Series(dtype=bool),
        threshold=0.0,
        model=None,
        scaler=MinMaxScaler(),
    )
    if not TF_AVAILABLE:
        return dummy_result

    data = feature_df.select_dtypes(include=[np.number]).dropna(axis=1).fillna(method="ffill").values
    if len(data) < seq_len * 2:
        logger.warning("Not enough data for LSTM Autoencoder.")
        return dummy_result

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X = _make_sequences(data_scaled, seq_len)
    n_features = data_scaled.shape[1]

    split = int(len(X) * 0.85)
    X_train, X_val = X[:split], X[split:]

    model = build_lstm_autoencoder(n_features, seq_len, latent_dim)
    logger.info("Training LSTM Autoencoder (%d sequences, %d features) …", len(X_train), n_features)

    model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[_early_stopping(20)],
        verbose=0,
    )

    # Compute reconstruction error on full dataset
    X_all = X
    recon = model.predict(X_all, verbose=0)
    mse_per_seq = np.mean(np.mean((X_all - recon) ** 2, axis=2), axis=1)

    # Align to original index: each sequence corresponds to its last time step
    idx = feature_df.dropna(axis=1).dropna().index
    idx_aligned = idx[seq_len - 1 :]
    if len(idx_aligned) > len(mse_per_seq):
        idx_aligned = idx_aligned[: len(mse_per_seq)]
    error_series = pd.Series(mse_per_seq[: len(idx_aligned)], index=idx_aligned, name="ae_reconstruction_error")

    # Threshold: mean + 2 std of training reconstruction errors
    train_errors = mse_per_seq[:split]
    threshold = float(np.mean(train_errors) + 2 * np.std(train_errors))
    anomaly_flag = error_series > threshold
    anomaly_flag.name = "ae_anomaly"

    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        model.save(save_path / "lstm_autoencoder.keras")

    logger.info(
        "LSTM Autoencoder: threshold=%.6f | anomalies=%.1f%%",
        threshold, anomaly_flag.mean() * 100,
    )
    return LSTMAutoencoderResult(
        reconstruction_error=error_series,
        anomaly_flag=anomaly_flag,
        threshold=threshold,
        model=model,
        scaler=scaler,
    )


# ============================================================
# 2.  LSTM Price Predictor
# ============================================================

@dataclass
class LSTMPredictorResult:
    predicted: pd.Series
    actual: pd.Series
    divergence: pd.Series          # |predicted - actual| normalised
    bubble_signal: pd.Series       # 0–1 bubble signal from divergence
    model: object
    scaler: object


def build_lstm_predictor(seq_len: int, n_features: int):
    if not TF_AVAILABLE:
        return None
    inputs = keras.Input(shape=(seq_len, n_features))
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1)(x)
    model = keras.Model(inputs, output, name="lstm_predictor")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def train_lstm_predictor(
    price_series: pd.Series,
    feature_df: pd.DataFrame,
    seq_len: int = 12,
    epochs: int = 100,
    batch_size: int = 16,
    save_path: Optional[Path] = None,
) -> LSTMPredictorResult:
    """
    Trains LSTM to predict next-period home price index from macro features.
    Large divergence between prediction and actual = anomalous price growth.
    """
    dummy = LSTMPredictorResult(
        predicted=pd.Series(dtype=float),
        actual=pd.Series(dtype=float),
        divergence=pd.Series(dtype=float),
        bubble_signal=pd.Series(dtype=float),
        model=None,
        scaler=MinMaxScaler(),
    )
    if not TF_AVAILABLE:
        return dummy

    combined = feature_df.copy()
    combined["_target"] = price_series
    combined = combined.select_dtypes(include=[np.number]).dropna(axis=1).fillna(method="ffill").dropna()
    if len(combined) < seq_len * 2 + 10:
        logger.warning("Not enough data for LSTM Predictor.")
        return dummy

    price_col_idx = combined.columns.get_loc("_target")
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(combined.values)

    X_seqs = _make_sequences(data_scaled, seq_len)
    y_seqs = data_scaled[seq_len:, price_col_idx]
    if len(X_seqs) > len(y_seqs):
        X_seqs = X_seqs[: len(y_seqs)]

    split = int(len(X_seqs) * 0.80)
    X_train, X_test = X_seqs[:split], X_seqs[split:]
    y_train, y_test = y_seqs[:split], y_seqs[split:]

    model = build_lstm_predictor(seq_len, data_scaled.shape[1])
    logger.info("Training LSTM Price Predictor …")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[_early_stopping(15)],
        verbose=0,
    )

    # Predictions on full dataset
    X_all = _make_sequences(data_scaled, seq_len)
    y_all = data_scaled[seq_len:, price_col_idx]
    if len(X_all) > len(y_all):
        X_all = X_all[: len(y_all)]
    preds_scaled = model.predict(X_all, verbose=0).flatten()

    # Invert scaling for target only
    dummy_matrix = np.zeros((len(preds_scaled), data_scaled.shape[1]))
    dummy_matrix[:, price_col_idx] = preds_scaled
    preds_orig = scaler.inverse_transform(dummy_matrix)[:, price_col_idx]
    dummy_matrix[:, price_col_idx] = y_all[: len(preds_scaled)]
    actuals_orig = scaler.inverse_transform(dummy_matrix)[:, price_col_idx]

    idx = combined.index[seq_len : seq_len + len(preds_orig)]
    predicted = pd.Series(preds_orig, index=idx, name="lstm_predicted")
    actual    = pd.Series(actuals_orig, index=idx, name="actual")
    divergence = (np.abs(predicted - actual) / actual.replace(0, np.nan)).rename("lstm_divergence")

    # Normalise divergence to 0-1 bubble signal
    d_min, d_max = divergence.min(), divergence.max()
    bubble_signal = ((divergence - d_min) / (d_max - d_min + 1e-8)).rename("lstm_bubble_signal")

    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        model.save(save_path / "lstm_predictor.keras")

    logger.info("LSTM Predictor: avg divergence=%.4f", divergence.mean())
    return LSTMPredictorResult(
        predicted=predicted,
        actual=actual,
        divergence=divergence,
        bubble_signal=bubble_signal,
        model=model,
        scaler=scaler,
    )


# ============================================================
# 3.  Feedforward Neural Network Classifier
# ============================================================

@dataclass
class FNNResult:
    bubble_prob: pd.Series
    accuracy: float
    auc: float
    model: object
    scaler: object


def build_fnn(n_features: int):
    if not TF_AVAILABLE:
        return None
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="fnn_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(5e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_fnn(
    X: pd.DataFrame,
    y: pd.Series,
    epochs: int = 200,
    batch_size: int = 32,
    save_path: Optional[Path] = None,
) -> FNNResult:
    """
    Feedforward Neural Network for bubble classification.
    Uses macro feature vector at each time step.
    """
    dummy = FNNResult(
        bubble_prob=pd.Series(dtype=float),
        accuracy=0.0,
        auc=0.0,
        model=None,
        scaler=MinMaxScaler(),
    )
    if not TF_AVAILABLE:
        return dummy

    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return dummy

    n = len(X)
    split = int(n * 0.80)
    X_train, X_test = X.iloc[:split].values, X.iloc[split:].values
    y_train, y_test = y.iloc[:split].values, y.iloc[split:].values

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Class weights for imbalance
    pos = y_train.sum()
    neg = len(y_train) - pos
    class_weight = {0: 1.0, 1: max(1.0, neg / (pos + 1))}

    model = build_fnn(X_train.shape[1])
    logger.info("Training FNN Classifier …")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[_early_stopping(20)],
        verbose=0,
    )

    y_prob_train = model.predict(scaler.transform(X.values), verbose=0).flatten()
    bubble_prob = pd.Series(y_prob_train, index=X.index, name="fnn_bubble_prob")

    y_pred_test = (model.predict(X_test, verbose=0).flatten() > 0.5).astype(int)
    y_prob_test  = model.predict(X_test, verbose=0).flatten()

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_pred_test)
    auc = roc_auc_score(y_test, y_prob_test) if len(np.unique(y_test)) > 1 else 0.5

    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        model.save(save_path / "fnn_classifier.keras")

    logger.info("FNN: acc=%.3f | AUC=%.3f", acc, auc)
    return FNNResult(
        bubble_prob=bubble_prob,
        accuracy=acc,
        auc=auc,
        model=model,
        scaler=scaler,
    )


# ============================================================
# Top-level runner
# ============================================================

def train_all_dl_models(
    feature_df: pd.DataFrame,
    price_series: pd.Series,
    X_labeled: pd.DataFrame,
    y_labeled: pd.Series,
    save_dir: Optional[Path] = None,
) -> dict:
    """Train all three deep learning models and return results dict."""
    if not TF_AVAILABLE:
        logger.warning("TensorFlow unavailable — skipping DL models.")
        return {"lstm_ae": None, "lstm_pred": None, "fnn": None}

    ae   = train_lstm_autoencoder(feature_df, save_path=save_dir)
    pred = train_lstm_predictor(price_series, feature_df, save_path=save_dir)
    fnn  = train_fnn(X_labeled, y_labeled, save_path=save_dir)

    return {"lstm_ae": ae, "lstm_pred": pred, "fnn": fnn}
