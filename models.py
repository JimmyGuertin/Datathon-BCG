import pandas as pd
import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# --- 1. Préparation des données ---
def prepare_data(df, features, targets, seq_length=168):
    """
    Normalisation et création de séquences pour LSTM.
    """
    df_model = df.copy()
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(df_model[features + targets])
    y_scaled = scaler_y.fit_transform(df_model[targets])
    
    X_seq, y_seq = [], []
    for i in range(seq_length, len(X_scaled)):
        X_seq.append(X_scaled[i-seq_length:i])
        y_seq.append(y_scaled[i])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # Split temporel train/test (pas aléatoire)
    train_size = int(len(X_seq) * 0.9)
    X_train, X_test = X_seq[:train_size], X_seq[train_size:]
    y_train, y_test = y_seq[:train_size], y_seq[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y

# --- 2. Création et entraînement du modèle ---
def train_lstm_24(X_train, y_train, X_test, y_test, lstm_units=64, dropout=0.2, epochs=40, batch_size=32):
    """
    Création d'un LSTM multi-target et entraînement.
    """
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(y_train.shape[1]))  # nombre de sorties = nombre de targets
    
    model.compile(optimizer='adam', loss='mse')
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        shuffle=False,
        verbose=1
    )
    
    return model, history

# --- 3. Évaluation du modèle et calcul du RMSE par target ---
def evaluate_model(model, X_test, y_test, scaler_y, targets):
    """
    Prédictions et RMSE pour chaque target.
    """
    y_pred = model.predict(X_test)
    
    y_test_inv = scaler_y.inverse_transform(y_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred)
    
    for i, t in enumerate(targets):
        rmse = np.sqrt(mean_squared_error(y_test_inv[:, i], y_pred_inv[:, i]))
        mean_val = np.mean(y_test_inv[:, i])
        print(f"✅ {t} : RMSE = {rmse:.2f}, Moyenne = {mean_val:.2f}, Erreur relative = {rmse/mean_val*100:.2f}%")
    
    return y_test_inv, y_pred_inv

# --- 4. Plot prédiction vs réel ---
def plot_predictions(y_test_inv, y_pred_inv, targets, n_plot=None):
    """
    Plot les targets réelles et prédites.
    n_plot = nombre de points à afficher (None pour tout)
    """
    if n_plot is not None:
        y_test_inv = y_test_inv[:n_plot]
        y_pred_inv = y_pred_inv[:n_plot]
    
    plt.figure(figsize=(15, 8))
    for i, t in enumerate(targets):
        plt.subplot(len(targets), 1, i+1)
        plt.plot(y_test_inv[:, i], label=f"{t} réel", marker='o', markersize=3)
        plt.plot(y_pred_inv[:, i], label=f"{t} prédit", marker='x', markersize=3)
        plt.title(f"Prédiction vs Réel : {t}")
        plt.xlabel("Index temporel")
        plt.ylabel(t)
        plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    
def train_lstm_seq2seq(df, features, targets, seq_length=168, pred_length=72,
                        epochs=20, batch_size=32, log_transform=False):
    """
    Entraîne un LSTM seq2seq pour prédire plusieurs cibles sur un horizon donné.
    
    Args:
        df (DataFrame): Dataframe contenant les features et targets
        features (list[str]): liste des colonnes features
        targets (list[str]): liste des colonnes targets
        seq_length (int): nombre de pas passés pour la séquence d'entrée
        pred_length (int): nombre de pas à prédire (horizon)
        epochs (int): nombre d'époques d'entraînement
        batch_size (int): taille du batch
        log_transform (bool): appliquer log1p sur les targets
    
    Returns:
        model: modèle LSTM entraîné
        history: historique Keras
        scaler_X: scaler pour les features
        scalers_y: liste de scalers pour les targets
        X_test, y_test: séquences de test pour évaluation
    """
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Lambda
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import Huber
    from sklearn.preprocessing import MinMaxScaler

    # Copie du dataframe
    df_model = df.copy()
    
    if log_transform:
        for target in targets:
            df_model[target] = np.log1p(df_model[target])

    # Normalisation
    scaler_X = MinMaxScaler()
    scalers_y = [MinMaxScaler() for _ in targets]
    y_scaled_cols = [scalers_y[i].fit_transform(df_model[[t]]) for i, t in enumerate(targets)]
    y_scaled = np.hstack(y_scaled_cols)  # shape=(n_samples, n_targets)

    # Normalisation features (incluant les targets si nécessaire)
    features_with_targets = features + targets
    X_scaled = scaler_X.fit_transform(df_model[features_with_targets])

    # Création des séquences seq2seq
    def create_sequences_seq2seq(X, y, seq_length, pred_length):
        X_seq, y_seq = [], []
        for i in range(seq_length, len(X) - pred_length + 1):
            X_seq.append(X[i-seq_length:i])
            y_seq.append(y[i:i+pred_length])
        return np.array(X_seq), np.array(y_seq)

    X_seq, y_seq = create_sequences_seq2seq(X_scaled, y_scaled, seq_length, pred_length)

    # Split temporel
    train_size = int(len(X_seq) * 0.9)
    X_train, X_test = X_seq[:train_size], X_seq[train_size:]
    y_train, y_test = y_seq[:train_size], y_seq[train_size:]

    # Définition du modèle seq2seq
    model = Sequential([
        LSTM(64, activation='tanh', input_shape=(seq_length, X_train.shape[2]), dropout=0.2),
        Dense(pred_length * len(targets)),
        Lambda(lambda x: tf.reshape(x, (-1, pred_length, len(targets))))
    ])
    model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0), loss=Huber())

    # Entraînement
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        shuffle=False,
        verbose=1
    )

    return model, history, scaler_X, scalers_y, X_test, y_test


import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate_lstm_seq2seq(model, X_test, y_test, scalers_y, targets, n_plot=72):
    """
    Évalue un modèle LSTM seq2seq sur le jeu de test.
    Affiche RMSE, moyenne et erreur relative pour chaque target.
    Trace les prédictions vs réel pour le premier échantillon.
    
    Args:
        model : modèle entraîné
        X_test, y_test : séquences de test
        scalers_y : liste de scalers pour chaque target
        targets : noms des targets
        n_plot : nombre de pas à afficher dans le plot (par défaut 72)
    """
    # Prédiction
    y_pred = model.predict(X_test)
    
    # Inverse la normalisation par target
    y_test_inv = np.zeros_like(y_test)
    y_pred_inv = np.zeros_like(y_pred)
    for i, scaler in enumerate(scalers_y):
        y_test_inv[:, :, i] = scaler.inverse_transform(y_test[:, :, i])
        y_pred_inv[:, :, i] = scaler.inverse_transform(y_pred[:, :, i])
    
    # Calcul métriques par target
    for i, t in enumerate(targets):
        rmse = np.sqrt(mean_squared_error(y_test_inv[:, :, i].flatten(), y_pred_inv[:, :, i].flatten()))
        mean_val = np.mean(y_test_inv[:, :, i])
        print(f"✅ {t} : RMSE = {rmse:.2f}, Moyenne = {mean_val:.2f}, Erreur relative = {rmse/mean_val*100:.2f}%")

    # Calcul métriques par target
    for i, t in enumerate(targets):
        # Sélection des 72 premières valeurs
        y_true = y_test_inv[:72, :, i].flatten()
        y_pred = y_pred_inv[:72, :, i].flatten()

        # Calcul des métriques
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mean_val = np.mean(y_true)
        print(f"✅ {t} : RMSE = {rmse:.2f}, Moyenne = {mean_val:.2f}, Erreur relative = {rmse/mean_val*100:.2f}%")
    
    # Plot premier échantillon
    plt.figure(figsize=(12, 6))
    for i, t in enumerate(targets):
        plt.subplot(len(targets), 1, i+1)
        plt.plot(y_test_inv[0,:n_plot, i], label=f"{t} réel", marker='o')
        plt.plot(y_pred_inv[0,:n_plot, i], label=f"{t} prédit", marker='x')
        plt.title(f"{t} : Prédictions vs Réel (premier échantillon)")
        plt.xlabel("Pas de temps")
        plt.ylabel(t)
        plt.legend()
        plt.tight_layout()
    plt.show()
    
    return y_test_inv, y_pred_inv
