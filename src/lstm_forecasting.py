import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

class LSTMForecaster:
    def __init__(self, seq_length=168, pred_length=24, lstm_units=64, dropout=0.2,
                use_weather=True, use_holidays=True, use_sport=True, use_outliers=True
                ):
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.features = None
        self.targets = ['Débit horaire', "Taux d'occupation"]
        # Feature selection
        self.use_weather = use_weather
        self.use_holidays = use_holidays
        self.use_sport = use_sport
        self.use_outliers = use_outliers

    def prepare_data(self, df):
        # Base time features
        self.features = [
            'hour_sin', 'hour_cos', 
            'weekday_sin', 'weekday_cos', 
            'month_sin', 'month_cos', 
            'dayofyear_sin', 'dayofyear_cos',
            'is_weekend'
        ]
        # Conditional features
        if self.use_holidays:
            self.features.append('is_holiday')
        if self.use_sport:
            self.features.append('is_sport_event')
        if self.use_outliers:
            self.features += ["Débit horaire_outlier_iqr", "Taux d'occupation_outlier_iqr"]
        if self.use_weather:
            self.features += [
                'temperature_2m (°C)', 
                'wind_speed_10m (km/h)',
                'precipitation (mm)', 
                'cloud_cover (%)'
            ]

        df = df.copy().sort_values('date')

        # Remove NaNs
        df = df[self.features + self.targets].dropna()

        # Normalisation
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        X_scaled = self.scaler_X.fit_transform(df[self.features])
        y_scaled = self.scaler_y.fit_transform(df[self.targets])

        X, y = [], []
        for i in range(len(df) - self.seq_length - self.pred_length):
            X.append(X_scaled[i:i+self.seq_length])
            y.append(y_scaled[i+self.seq_length:i+self.seq_length+self.pred_length])

        X, y = np.array(X), np.array(y)

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        return X_train, X_test, y_train, y_test

    def build_model(self, n_features, n_targets):
        model = Sequential([
            LSTM(self.lstm_units, input_shape=(self.seq_length, n_features), return_sequences=False),
            Dropout(self.dropout),
            Dense(self.pred_length * n_targets)
        ])
        model.compile(optimizer='adam', loss='mse')
        self.model = model
        return model

    def train(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
        n_features = X_train.shape[2]
        n_targets = len(self.targets)
        self.build_model(n_features, n_targets)

        es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        history = self.model.fit(
            X_train, y_train.reshape(len(y_train), -1),
            validation_data=(X_test, y_test.reshape(len(y_test), -1)),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=1
        )
        return history
    
    def predict(self, X):
        """
        Predict from prepared array (X) and return inverse-transformed results.
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        y_pred_scaled = self.model.predict(X)
        n_targets = len(self.targets)
        y_pred_inv = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, n_targets))
        y_pred_inv = y_pred_inv.reshape(X.shape[0], self.pred_length, n_targets)
        return y_pred_inv

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance using RMSE and relative error.
        """
        y_pred_inv = self.predict(X_test)
        n_targets = len(self.targets)
        y_test_inv = self.scaler_y.inverse_transform(y_test.reshape(-1, n_targets)).reshape(y_test.shape)

        for i, target in enumerate(self.targets):
            rmse = np.sqrt(np.mean((y_pred_inv[:,:,i] - y_test_inv[:,:,i])**2))
            mean_val = np.mean(y_test_inv[:,:,i])
            rel_err = 100 * rmse / mean_val
            print(f"{target} : RMSE = {rmse:.2f}, Mean = {mean_val:.2f}, Relative error = {rel_err:.2f}%")

        return y_test_inv, y_pred_inv

    def plot_predictions(self, y_test_inv, y_pred_inv, n_plot=72):
        n_targets = len(self.targets)
        # Create one subplot per target stacked vertically
        fig, axes = plt.subplots(n_targets, 1, figsize=(15, 4 * n_targets), squeeze=False)

        for i, target in enumerate(self.targets):
            ax = axes[i, 0]
            ax.plot(y_test_inv[:n_plot, 0, i], label=f"True {target}", linewidth=2)
            ax.plot(y_pred_inv[:n_plot, 0, i], '--', label=f"Predicted {target}")
            ax.set_title(f"Forecasting - {target}")
            ax.set_xlabel("Hour (index)")
            ax.legend()

        plt.tight_layout()
        plt.show()
    
    def time_series_cv(self, X, y, n_splits=5, epochs=10, batch_size=32):
        """
        Cross-validation
        
        Args:
            X, y : All sequences
            n_splits : number of temporal splits
            epochs, batch_size : parameters
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        rmse_list = []
        rel_error_list = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Réinitialisation du modèle
            self.model = None
            self.train(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size)

            y_test_inv, y_pred_inv = self.evaluate(X_test, y_test)

            rmse_split = []
            rel_error_split = []
            for i in range(len(self.targets)):
                rmse_i = np.sqrt(np.mean((y_pred_inv[:,:,i] - y_test_inv[:,:,i])**2))
                mean_val = np.mean(y_test_inv[:,:,i])
                rel_error_i = 100 * rmse_i / mean_val
                rmse_split.append(rmse_i)
                rel_error_split.append(rel_error_i)
            rmse_list.append(rmse_split)
            rel_error_list.append(rel_error_split)

        rmse_array = np.array(rmse_list)
        rel_error_array = np.array(rel_error_list)

        # Affichage des moyennes et écarts-types
        for i, target in enumerate(self.targets):
            print(f"{target} : RMSE CV mean = {rmse_array[:,i].mean():.2f} ± {rmse_array[:,i].std():.2f}, "
                  f"RelError CV mean = {rel_error_array[:,i].mean():.2f}% ± {rel_error_array[:,i].std():.2f}%")
