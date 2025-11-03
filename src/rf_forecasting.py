# src/rf_forecasting.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


class RandomForestForecaster:
    """
    Forecaster basé sur RandomForest, avec une API proche de LSTMForecaster :
      - prepare_data(df)
      - train(X_train, y_train, X_val=None, y_val=None)
      - evaluate(X_test, y_test)
      - plot_predictions(y_test_df, y_pred_df, n_plot=72)
    """

    def __init__(
        self,
        target_cols=None,
        lags=None,
        rolling_windows=None,
        train_ratio=0.8,
        rf_params=None,
    ):
        # Cibles par défaut : à adapter si besoin
        self.target_cols = target_cols or ["Débit horaire", "Taux d'occupation"]

        # Lags en heures
        self.lags = lags or [1, 2, 3, 6, 24]

        # Fenêtres pour moyennes glissantes
        self.rolling_windows = rolling_windows or [6, 24]

        # Ratio train / test (split temporel)
        self.train_ratio = train_ratio

        # Hyperparamètres du RF
        self.rf_params = rf_params or {
            "n_estimators": 400,
            "max_depth": None,
            "min_samples_leaf": 3,
            "max_features": "sqrt",
            "n_jobs": -1,
            "random_state": 42,
        }

        # Objets appris
        self.models_ = {}          # un modèle par cible
        self.feature_names_ = None
        self.train_index_ = None
        self.test_index_ = None

    # ------------------------------------------------------------------ #
    # Préparation des données : features tabulaires + split temporel
    # ------------------------------------------------------------------ #

    def _add_time_features(self, df):
        """
        Ajoute des features temporelles (hour, dow, month, encodage sin/cos)
        en supposant que l'index est de type datetime.
        """
        df = df.copy()
        if not np.issubdtype(df.index.dtype, np.datetime64):
            raise ValueError(
                "L'index du DataFrame doit être de type datetime pour ajouter "
                "les features temporelles."
            )

        df["hour"] = df.index.hour
        df["dow"] = df.index.dayofweek
        df["month"] = df.index.month

        # Encodage cyclique
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

        return df

    def _add_lags_and_rolling(self, df):
        """
        Ajoute des lags et des moyennes glissantes pour les colonnes cibles.
        """
        df = df.copy()
        for col in self.target_cols:
            for lag in self.lags:
                df[f"{col}_lag_{lag}h"] = df[col].shift(lag)

            for win in self.rolling_windows:
                df[f"{col}_rollmean_{win}h"] = df[col].rolling(window=win).mean()

        # Suppression des lignes avec NaN introduits par shift / rolling
        df = df.dropna()
        return df

    def prepare_data(self, df):
        print(f"[RF] input df shape: {df.shape}")
        df = df.copy()

        # s'assurer d'un index datetime
        if not np.issubdtype(df.index.dtype, np.datetime64):
            if "Date et heure de comptage" in df.columns:
                df["Date et heure de comptage"] = pd.to_datetime(df["Date et heure de comptage"], errors="coerce")
                df = df.set_index("Date et heure de comptage")
            else:
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    pass
        df = df.sort_index()

        # checks
        max_lag = max(getattr(self, "lags", [0])) if getattr(self, "lags", None) else 0
        max_roll = max(getattr(self, "rolling_windows", [0])) if getattr(self, "rolling_windows", None) else 0
        required_rows = max_lag + max_roll
        print(f"[RF] max_lag={max_lag} max_roll={max_roll} required_rows={required_rows} len(df)={len(df)}")
        if len(df) <= required_rows:
            raise ValueError(f"[RF] Pas assez de lignes: {len(df)} <= required_rows ({required_rows}). Réduisez lags/rolling ou ajoutez des données.")

        features = df.copy()

        # création des lags
        for lag in getattr(self, "lags", []):
            for col in self.target_cols:
                features[f"{col}_lag_{lag}"] = features[col].shift(lag)

        # rolling avec min_periods=1 pour limiter NaN
        for win in getattr(self, "rolling_windows", []):
            for col in self.target_cols:
                features[f"{col}_roll_mean_{win}"] = features[col].rolling(window=win, min_periods=1).mean()

        # NE DROPPEZ QUE LES LIGNES OÙ LA/LES CIBLES SONT NaN
        before_shape = features.shape
        features = features.dropna(subset=self.target_cols, how="any")
        print(f"[RF] shape before dropping target-NaN: {before_shape} -> after: {features.shape}")

        # Supprimer colonnes trop creuses (ex : >50% NaN)
        thresh = int(0.5 * len(features))
        features = features.dropna(axis=1, thresh=thresh)
        print(f"[RF] shape after dropping sparse columns: {features.shape}")

        # utiliser ffill / bfill (évite FutureWarning)
        features = features.ffill().bfill()
        # pour les colonnes numériques, remplir les NaN restants par 0
        num_cols = features.select_dtypes(include=[np.number]).columns
        features[num_cols] = features[num_cols].fillna(0)
    
        # Sélectionner X = features numériques (évite l'erreur "could not convert string to float")
        y = features[self.target_cols].copy()
        X = features.drop(columns=self.target_cols, errors="ignore")

        # --- FORCER suppression / encodage des colonnes non numériques avant le split ---
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            print(f"[RF] dropping non-numeric columns before finalizing X: {non_numeric}")
            # Option A = DROP (rapide, conserve uniquement numériques)
            X = X.select_dtypes(include=[np.number])

            # Option B = ONE-HOT ENCODING (décommenter si vous voulez garder l'information)
            # cat_cols = non_numeric
            # print(f"[RF] one-hot encoding plutôt que drop: {cat_cols}")
            # X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dummy_na=False)
        # --- fin suppression/encodage ---

        print(f"[RF] final X shape: {X.shape}, y shape: {y.shape}")

        # split temporel
        n = len(X)
        train_size = int(n * getattr(self, "train_ratio", 0.8))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        print(f"[RF] X_train: {X_train.shape}, X_test: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------ #
    # Entraînement
    # ------------------------------------------------------------------ #

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Entraîne un RandomForestRegressor PAR cible.
        (X_val / y_val sont ignorés, juste pour compatibilité d'API.)
        """
        if X_train is None or X_train.shape[0] == 0:
            raise ValueError("[RF] X_train est vide après préparation. Vérifiez prepare_data / réduisez lags/rolling / ajoutez des données.")
        
        self.models_ = {}

        for col in self.target_cols:
            model = RandomForestRegressor(**self.rf_params)
            model.fit(X_train, y_train[col])
            self.models_[col] = model

        print(f"[RF] Entraînement terminé pour cibles : {self.target_cols}")
        return self.models_

    # ------------------------------------------------------------------ #
    # Évaluation
    # ------------------------------------------------------------------ #

    def evaluate(self, X_test, y_test):
        """
        Prédit sur X_test, calcule les métriques et retourne :
          - y_test_df (DataFrame)
          - y_pred_df (DataFrame)
        """
        if not self.models_:
            raise RuntimeError("Les modèles RandomForest n'ont pas encore été entraînés.")

        y_pred = pd.DataFrame(index=y_test.index, columns=self.target_cols, dtype=float)

        for col in self.target_cols:
            model = self.models_[col]
            y_pred[col] = model.predict(X_test)

        # Calcul des métriques
        for col in self.target_cols:
            rmse = np.sqrt(mean_squared_error(y_test[col], y_pred[col]))
            mean_val = y_test[col].mean()
            rel_err = rmse / mean_val * 100 if mean_val != 0 else np.nan

            print(
                f"{col} : RMSE = {rmse:.2f}, "
                f"Mean = {mean_val:.2f}, "
                f"Relative error = {rel_err:.2f}%"
            )

        return y_test, y_pred

    # ------------------------------------------------------------------ #
    # Visualisation
    # ------------------------------------------------------------------ #

    def plot_predictions(self, y_test_df, y_pred_df, n_plot=72):
        """
        Visualise les n_plot premières heures pour chaque cible.
        y_test_df et y_pred_df sont les DataFrames retournés par evaluate().
        """
        for col in self.target_cols:
            plt.figure(figsize=(12, 4))
            true_series = y_test_df[col].iloc[:n_plot]
            pred_series = y_pred_df[col].iloc[:n_plot]

            plt.plot(true_series.index, true_series.values, label="Vrai", linewidth=2)
            plt.plot(pred_series.index, pred_series.values, label="Prédit (RF)", linestyle="--")

            plt.title(f"Random Forest – {col} (premières {n_plot} heures)")
            plt.xlabel("Date")
            plt.ylabel(col)
            plt.legend()
            plt.tight_layout()
            plt.show()

    def cross_validate(self, X, y, n_splits=5, refit=True, random_state=None):
        """
        Time-series cross-validation using TimeSeriesSplit.
        - X : DataFrame des features (index temporel)
        - y : DataFrame des cibles (multi-colonnes possible)
        - n_splits : nombre de folds TimeSeriesSplit
        - refit : si True, entraîne des modèles finaux sur l'ensemble X/y après CV
        Retourne un dict de listes de RMSE par cible.
        """
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error

        if X is None or y is None:
            raise ValueError("[RF CV] X et y doivent être fournis")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = {col: [] for col in self.target_cols}

        fold = 0
        for train_idx, val_idx in tscv.split(X):
            fold += 1
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            print(f"[RF CV] fold {fold}: train={X_tr.shape[0]} val={X_val.shape[0]}")

            for col in self.target_cols:
                model = RandomForestRegressor(**{**self.rf_params, **({'random_state': random_state} if random_state is not None else {})})
                model.fit(X_tr, y_tr[col])
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val[col], y_pred))
                scores[col].append(rmse)

        # affichage résumé
        for col, vals in scores.items():
            arr = np.array(vals)
            print(f"[RF CV] {col} RMSE per fold: {np.round(arr, 3).tolist()} mean={arr.mean():.3f} std={arr.std():.3f}")

        # refit final sur tout X/y si demandé
        if refit:
            self.models_ = {}
            for col in self.target_cols:
                m = RandomForestRegressor(**{**self.rf_params, **({'random_state': random_state} if random_state is not None else {})})
                m.fit(X, y[col])
                self.models_[col] = m
            print("[RF CV] final models refit on full data")

        return scores
