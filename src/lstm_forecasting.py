# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

##### NEW LSTM
class LSTMTimeSeries:
    def __init__(self, features, targets, seq_length=168, pred_length=72):
        self.features = features
        self.targets = targets
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.model_final = None

    # -----------------------------
    # Séquence multi-step
    # -----------------------------
    def create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(len(X) - self.seq_length - self.pred_length + 1):
            X_seq.append(X[i:i+self.seq_length])
            y_seq.append(y[i+self.seq_length:i+self.seq_length+self.pred_length])
        return np.array(X_seq), np.array(y_seq)
    
    # -----------------------------
    # Métriques
    # -----------------------------
    @staticmethod
    def evaluate_metrics(y_true, y_pred, target_name=None):
        """
        Calcule les métriques pour une seule target.
        Arguments :
            y_true : array (n_samples,)
            y_pred : array (n_samples,)
            target_name : str (optionnel, pour affichage)
        Retourne : dict contenant RMSE, MAPE, RMSE/mean, mean
        """

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred)
        true_mean = np.mean(y_true)
        rmse_mean = rmse / true_mean

        if target_name:
            print(f"{target_name:<25} RMSE: {rmse:.2f}, MEAN: {true_mean:.2f}, "
                f"MAPE: {mape:.2%}, RMSE/mean: {rmse_mean:.2%}")

        return {"target": target_name,
            "rmse": rmse,
            "mean": true_mean,
            "mape": mape,
            "rmse_mean": rmse_mean}


    # -----------------------------
    # Normalisation
    # -----------------------------
    def scale_data(self, df):
        X_scaled = self.scaler_X.fit_transform(df[self.features])
        y_scaled = self.scaler_y.fit_transform(df[self.targets])
        return X_scaled, y_scaled

    # -----------------------------
    # Cross-validation temporelle
    # -----------------------------
    def cross_validate(self, df, df_original=None, n_splits=5, epochs=20, batch_size=32):
        """
        Cross-validation temporelle sur les séquences LSTM.
        
        Arguments :
            df : pd.DataFrame, données éventuellement lissées pour l'entraînement
            df_original : pd.DataFrame, données originales pour l'évaluation (sans rolling mean)
            n_splits : int, nombre de folds
            epochs : int
            batch_size : int
        """
        X_scaled, y_scaled = self.scale_data(df)
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
        if df_original is not None:
        # Créer les mêmes séquences sur les données originales
            X_orig, y_orig = self.create_sequences(df_original[self.features].values,
                                               df_original[self.targets].values)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_seq)):
            X_train, X_val = X_seq[train_idx], X_seq[val_idx]
            y_train, y_val = y_seq[train_idx], y_seq[val_idx]
            
            self.model = Sequential()
            self.model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dense(self.pred_length * len(self.targets)))
            self.model.compile(optimizer='adam', loss='mse', metrics=['mape','mae'])
            
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            self.model.fit(
                X_train, y_train.reshape(y_train.shape[0], -1),
                validation_data=(X_val, y_val.reshape(y_val.shape[0], -1)),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=0
            )

            # Prédiction
            y_val_pred_scaled = self.model.predict(X_val).reshape(y_val.shape)
            y_val_pred = self.scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, len(self.targets)))

            # ---- Choisir les vraies valeurs pour l'évaluation ----
            # Utiliser les séquences originales pour la vraie évaluation
            if df_original is not None:
                y_val_true = y_orig[val_idx].reshape(-1, len(self.targets))
            else:
                y_val_true = self.scaler_y.inverse_transform(y_val.reshape(-1, len(self.targets)))

            # Évaluation par target
            metrics_per_target = []
            print(f"\n===== Fold {fold+1} =====")
            for i, target_name in enumerate(self.targets):
                metrics = self.evaluate_metrics(
                    y_val_true[:, i], y_val_pred[:, i],
                    target_name=target_name
                )
                metrics_per_target.append(metrics)


    # -----------------------------
    # Entraînement final sur toutes les données
    # -----------------------------
    def train_final(self, df, epochs=30, batch_size=32):
        X_scaled, y_scaled = self.scale_data(df)
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
        
        self.model_final = Sequential()
        self.model_final.add(LSTM(128, input_shape=(self.seq_length, len(self.features))))
        self.model_final.add(Dropout(0.2))
        self.model_final.add(Dense(64, activation='relu'))
        self.model_final.add(Dense(self.pred_length * len(self.targets)))
        self.model_final.compile(optimizer='adam', loss='mse', metrics=['mape','mae'])
        
        self.model_final.fit(
            X_seq, y_seq.reshape(y_seq.shape[0], -1),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)],
            verbose=1
        )

    # -----------------------------
    # Prédiction sur nouvelle séquence
    # -----------------------------
    def predict(self, df_last):
        X_last_scaled = self.scaler_X.transform(df_last[self.features])
        X_last_seq = X_last_scaled[-self.seq_length:].reshape(1, self.seq_length, len(self.features))
        y_pred_scaled = self.model_final.predict(X_last_seq).reshape(self.pred_length, len(self.targets))
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return y_pred
