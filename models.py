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
    # --- Définir la taille du test alignée sur la saisonnalité ---
    test_size = int(len(X_seq) * 0.1)  # 10% des séquences
    # Ajuster pour avoir un multiple de 24
    test_size_final = test_size + (24 - test_size % 24) if test_size % 24 != 0 else test_size

    # --- Calcul du train_size aligné sur la saisonnalité ---
    train_size_final = len(X_seq) - test_size_final
    # Ajuster train_size pour que ce soit multiple de 24
    train_size_final -= train_size_final % 24

    # --- Découpage train/test ---
    X_train, X_test = X_seq[:train_size_final], X_seq[train_size_final:train_size_final + test_size_final]
    y_train, y_test = y_seq[:train_size_final], y_seq[train_size_final:train_size_final + test_size_final]

    # Définition du modèle seq2seq
    model = Sequential([
        LSTM(64, input_shape=(seq_length, X_train.shape[2]), dropout=0.2),
        Dense(pred_length * len(targets)),
        Lambda(lambda x: tf.reshape(x, (-1, pred_length, len(targets))))
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

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
from sklearn.metrics import r2_score

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
        r2 = r2_score(y_true, y_pred)
        print("R² :", r2)
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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- 1. Préparation des données ---
def prepare_data_multistep(df, features, targets, seq_length=168, pred_length=24):
    """
    Prépare les données pour une prédiction multi-step (ex: 24h d'avance)
    - seq_length : longueur de la séquence d'entrée (ex: 168h)
    - pred_length : longueur de la séquence de sortie (ex: 24h)
    """
    df_model = df.copy()

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # On normalise les colonnes features et targets
    X_scaled = scaler_X.fit_transform(df_model[features + targets])
    y_scaled = scaler_y.fit_transform(df_model[targets])

    X_seq, y_seq = [], []
    for i in range(seq_length, len(X_scaled) - pred_length + 1):
        X_seq.append(X_scaled[i - seq_length:i])
        y_seq.append(y_scaled[i:i + pred_length])  # plusieurs pas de sortie

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Split temporel (pas aléatoire)
    train_size = int(len(X_seq) * 0.9)
    X_train, X_test = X_seq[:train_size], X_seq[train_size:]
    y_train, y_test = y_seq[:train_size], y_seq[train_size:]

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y


# --- 2. Création et entraînement du modèle ---
def train_lstm_multistep(X_train, y_train, X_test, y_test, lstm_units=64, dropout=0.2, epochs=40, batch_size=32):
    """
    Modèle LSTM multi-output : prédit plusieurs pas à la fois (multi-step).
    """
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(y_train.shape[1] * y_train.shape[2]))  # sortie = nb_pas × nb_cibles
    model.compile(optimizer='adam', loss='mse')

    # Entraînement
    history = model.fit(
        X_train, 
        y_train.reshape(y_train.shape[0], -1),  # aplatir pour correspondre à la Dense
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test.reshape(y_test.shape[0], -1)),
        shuffle=False,
        verbose=1
    )

    return model, history


# --- 3. Évaluation du modèle et calcul du RMSE par target ---
def evaluate_model_multistep(model, X_test, y_test, scaler_y, targets, pred_length):
    """
    Évalue le modèle multi-step et affiche le RMSE moyen sur les horizons de prédiction.
    """
    y_pred = model.predict(X_test)
    n_targets = len(targets)

    # Reshape pour retrouver la structure (échantillons, pred_length, targets)
    y_pred = y_pred.reshape(-1, pred_length, n_targets)

    # Inverse scaling
    y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, n_targets)).reshape(y_test.shape)
    y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, n_targets)).reshape(y_pred.shape)

    for i, t in enumerate(targets):
        rmse = np.sqrt(mean_squared_error(y_test_inv[:, :, i].ravel(), y_pred_inv[:, :, i].ravel()))
        mean_val = np.mean(y_test_inv[:, :, i])
        print(f"✅ {t} : RMSE = {rmse:.2f}, Moyenne = {mean_val:.2f}, Erreur relative = {rmse / mean_val * 100:.2f}%")

    return y_test_inv, y_pred_inv


# --- 4. Plot des prédictions multi-step ---
def plot_predictions_multistep(y_test_inv, y_pred_inv, targets, step=0, n_plot=None):
    """
    Affiche les prédictions sur une fenêtre donnée (step = index de séquence test)
    n_plot = nombre de pas à afficher (None pour tout)
    """
    plt.figure(figsize=(15, 8))

    for i, t in enumerate(targets):
        plt.subplot(len(targets), 1, i + 1)
        y_real = y_test_inv[step, :n_plot, i]
        y_forecast = y_pred_inv[step, :n_plot, i]

        plt.plot(y_real, label=f"{t} réel", marker='o', markersize=3)
        plt.plot(y_forecast, label=f"{t} prédit", marker='x', markersize=3)
        plt.title(f"Prédiction sur {len(y_real)} pas : {t}")
        plt.xlabel("Horizon (pas de temps futurs)")
        plt.ylabel(t)
        plt.legend()

    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. Préparation des données multi-step ---
def prepare_data_time_series(df, features, targets, seq_length=168, pred_length=24):
    """
    Crée des séquences X et y pour LSTM multi-step.
    """
    df_model = df.copy()
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(df_model[features + targets])
    y_scaled = scaler_y.fit_transform(df_model[targets])

    X_seq, y_seq = [], []
    for i in range(seq_length, len(X_scaled) - pred_length + 1):
        X_seq.append(X_scaled[i - seq_length:i])
        y_seq.append(y_scaled[i:i + pred_length])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    return X_seq, y_seq, scaler_X, scaler_y

# --- 2. Création du modèle LSTM ---
def build_lstm_model_time_series(input_shape, output_dim, lstm_units=64, dropout=0.2):
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(output_dim))
    model.compile(optimizer='adam', loss='mse')
    return model

# --- 3. TimeSeriesSplit aligné sur saisonnalité ---
def train_evaluate_timeseries_split(df_lisse, features, targets, seq_length=168, pred_length=24,
                                    season_len=24, n_splits=5, lstm_units=64, dropout=0.2,
                                    epochs=40, batch_size=32):
    """
    Entraîne et évalue un LSTM multi-step en utilisant un TimeSeriesSplit
    qui respecte la saisonnalité (ex: 24 pas = 1 jour).
    Les tests utilisent systématiquement les dernières données.
    """
    # Préparation des séquences
    X, y, scaler_X, scaler_y = prepare_data_time_series(df_lisse, features, targets, seq_length, pred_length)
    n_targets = len(targets)
    rmse_results = {t: [] for t in targets}

    # Tronquer pour des multiples de season_len
    n_samples = len(X) - (len(X) % season_len)
    X, y = X[:n_samples], y[:n_samples]

    # Taille du test
    test_size = season_len * max(1, n_samples // (n_splits * season_len))

    print(f"Total sequences: {n_samples}, Test size: {test_size} ({test_size // season_len} jours)")

    # Boucle sur les splits (dernières fenêtres pour test)
    split_idx=0
    test_end = n_samples - split_idx * test_size
    test_start = test_end - test_size
    train_end = test_start

    # Ajuster pour avoir des multiples de season_len
    if train_end % season_len != 0:
        train_end -= train_end % season_len

    X_train, X_test = X[:train_end], X[test_start:test_end]
    y_train, y_test = y[:train_end], y[test_start:test_end]

    print(f"\nSplit {split_idx+1}/{n_splits}: Train={len(X_train)} ({len(X_train)//season_len}j), "
            f"Test={len(X_test)} ({len(X_test)//season_len}j)")

    # Création et entraînement du modèle
    model = build_lstm_model_time_series(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        output_dim=y_train.shape[1]*y_train.shape[2],
        lstm_units=lstm_units,
        dropout=dropout
    )

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train.reshape(len(y_train), -1),
        validation_data=(X_test, y_test.reshape(len(y_test), -1)),
        epochs=epochs, batch_size=batch_size, shuffle=False,
        verbose=0, callbacks=[es]
    )

    # Prédictions
    y_pred = model.predict(X_test, verbose=0).reshape(-1, pred_length, n_targets)
    y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, n_targets)).reshape(y_test.shape)
    y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, n_targets)).reshape(y_pred.shape)

    # Calcul RMSE
    for i, t in enumerate(targets):
        rmse = np.sqrt(mean_squared_error(y_test_inv[:,:,i].ravel(), y_pred_inv[:,:,i].ravel()))
        rmse_results[t].append(rmse)
        print(f"{t} split {split_idx+1}: RMSE={rmse:.2f}")

    # Résumé
    print("\nRésumé final:")
    for t in targets:
        print(f"{t}: moyenne RMSE = {np.mean(rmse_results[t]):.2f} ± {np.std(rmse_results[t]):.2f}")

    return model, rmse_results, (y_test_inv, y_pred_inv, scaler_y)


