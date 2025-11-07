import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

class XGBoostModel:
    def __init__(self, df: pd.DataFrame, features: list, targets: list, log_transform: bool=False):
        self.df = df.copy()
        self.features = features
        self.targets = targets
        self.log_transform = log_transform
        self.models = {}  # one model per target
        self.scaler_X = None
        self.scalers_y = {}
    
    def prepare_data(self, horizon=72):
        """
        Data preparation: scaling, log-transform (if needed), train/test split
        """
        df = self.df.dropna(subset=self.features + self.targets).copy()
        
        X = df[self.features].values
        y = df[self.targets].values
        
        # Scaling features
        self.scaler_X = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X)
        
        # Scaling targets and log-transform if needed
        y_scaled = np.zeros_like(y)
        for i, t in enumerate(self.targets):
            scaler_y = StandardScaler()
            y_scaled[:, i] = scaler_y.fit_transform(y[:, i].reshape(-1,1)).ravel()
            self.scalers_y[t] = scaler_y
            if self.log_transform:
                y_scaled[:, i] = np.log1p(y[:, i])
        
        # Split train/test by last 'horizon' hours
        split_idx = len(df) - horizon
        self.X_train, self.X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        self.y_train, self.y_test = y_scaled[:split_idx], y_scaled[split_idx:]
        self.df_test = df.iloc[split_idx:]

        print(f"Train set: {self.X_train.shape}, Test set: {self.X_test.shape} ({horizon} hours)")


    
    def fit(self, **xgb_params):
        """
        Fit an XGBoost model for each target
        """
        for i, t in enumerate(self.targets):
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                **xgb_params
            )
            model.fit(self.X_train, self.y_train[:, i])
            self.models[t] = model
    
    def predict(self, X=None):
        """
        Return descaled predictions for the test set or provided X
        """
        if X is None:
            X = self.X_test
        y_pred = np.zeros((X.shape[0], len(self.targets)))
        for i, t in enumerate(self.targets):
            pred = self.models[t].predict(X)
            if self.log_transform:
                pred = np.expm1(pred)
            else:
                pred = self.scalers_y[t].inverse_transform(pred.reshape(-1,1)).ravel()
            y_pred[:, i] = pred
        return y_pred
    
    def evaluate(self):
        """
        Evaluate RMSE and relative error on the test set
        """
        y_pred = self.predict()

        y_true_list = [self.df_test[t].values for t in self.targets]
        y_true_array = np.stack(y_true_list, axis=1)  # shape (n_samples, n_targets)
        
        results = {}
        for i, t in enumerate(self.targets):
            y_true = self.df_test[t].values
            rmse = np.sqrt(mean_squared_error(y_true, y_pred[:, i]))
            mean_val = y_true.mean()
            rel_error = 100 * rmse / mean_val
            results[t] = {'RMSE': rmse, 'Mean': mean_val, 'RelError(%)': rel_error}
            print(f"{t} : RMSE = {rmse:.2f}, Mean = {mean_val:.2f}, Relative Error = {rel_error:.2f}%")
        return y_true_array, y_pred
    

    def cross_validate(self, n_splits=5, **xgb_params):
        """
        TimeSeries cross-validation with RMSE, MAE, and MAPE evaluation
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        X = self.scaler_X.transform(self.df[self.features].values)
        y = np.column_stack([self.scalers_y[t].transform(self.df[[t]]) for t in self.targets])

        cv_results = {t: {'rmse': [], 'mae': [], 'mape': [], 'RMSE/MEAN':[]} for t in self.targets}

        for train_idx, test_idx in tscv.split(X):
            X_train_cv, X_test_cv = X[train_idx], X[test_idx]
            y_train_cv, y_test_cv = y[train_idx], y[test_idx]
            
            for i, t in enumerate(self.targets):
                model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    **xgb_params
                )
                model.fit(X_train_cv, y_train_cv[:, i])
                y_pred_cv = model.predict(X_test_cv)
                y_pred_cv = self.scalers_y[t].inverse_transform(y_pred_cv.reshape(-1,1)).ravel()
                y_true_cv = self.scalers_y[t].inverse_transform(y_test_cv[:, i].reshape(-1,1)).ravel()

                mean_true = np.mean(y_true_cv)                
                rmse = np.sqrt(mean_squared_error(y_true_cv, y_pred_cv))
                rmse_mean = rmse/mean_true*100
                mae = mean_absolute_error(y_true_cv, y_pred_cv)
                mape = mean_absolute_percentage_error(y_true_cv, y_pred_cv) * 100  # en pourcentage

                cv_results[t]['rmse'].append(rmse)
                cv_results[t]['mae'].append(mae)
                cv_results[t]['mape'].append(mape)
                cv_results[t]['RMSE/MEAN'].append(rmse_mean)

        # Display
        for t in self.targets:
            rmses = cv_results[t]['rmse']
            maes = cv_results[t]['mae']
            mapes = cv_results[t]['mape']
            RMSE_mean = cv_results[t]['RMSE/MEAN']
            print(f"{t} : CV RMSE mean = {np.mean(rmses):.2f}, std = {np.std(rmses):.2f} | "
                f"MAE mean = {np.mean(maes):.2f}, std = {np.std(maes):.2f} | "
                f"\n MAPE mean = {np.mean(mapes):.2f}%, std = {np.std(mapes):.2f}% | "
                f"RMSE/mean = {np.mean(RMSE_mean):.2f}%, std = {np.std(RMSE_mean):.2f}")

        return cv_results
    

    def full_train(self, **xgb_params):
        """
        Fit XGBoost models on the full dataset (no train/test split).
        Returns the trained models dictionary.
        """
        df = self.df.dropna(subset=self.features + self.targets).copy()
        
        X = df[self.features].values
        y = df[self.targets].values

        # Scaling features
        self.scaler_X = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X)

        # Scaling targets and log-transform if needed
        y_scaled = np.zeros_like(y)
        for i, t in enumerate(self.targets):
            scaler_y = StandardScaler()
            y_scaled[:, i] = scaler_y.fit_transform(y[:, i].reshape(-1,1)).ravel()
            self.scalers_y[t] = scaler_y
            if self.log_transform:
                y_scaled[:, i] = np.log1p(y[:, i])

        # Fit a model for each target
        self.models = {}
        for i, t in enumerate(self.targets):
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                **xgb_params
            )
            model.fit(X_scaled, y_scaled[:, i])
            self.models[t] = model

        print("Full training completed on all data.")
        return self.models

    
    def predict_final(self, df):
        """
        Predict on a provided DataFrame using trained models.
        The DataFrame must contain all required features.
        Returns descaled predictions as a numpy array.
        """
        if not self.models:
            raise ValueError("Models are not trained yet. Run full_train() first.")
       
        # Vérifie que toutes les features nécessaires sont bien présentes
        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features in provided DataFrame: {missing_features}")
 
        # Nettoyage et préparation
        df_input = df.dropna(subset=self.features).copy()
        X = df_input[self.features].values
        X_scaled = self.scaler_X.transform(X)
 
        # Prédictions
        y_pred = np.zeros((X.shape[0], len(self.targets)))
        for i, t in enumerate(self.targets):
            pred = self.models[t].predict(X_scaled)
            if self.log_transform:
                pred = np.expm1(pred)
            else:
                pred = self.scalers_y[t].inverse_transform(pred.reshape(-1,1)).ravel()
            y_pred[:, i] = pred
 
        # Créer le DataFrame final avec uniquement date, hour et prédictions
        df_pred = pd.DataFrame()
        
        for i, t in enumerate(self.targets):
            df_pred[f"pred_{t}"] = y_pred[:, i]

        return df_pred
        
 