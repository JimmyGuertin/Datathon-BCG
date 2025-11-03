import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
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
        TimeSeries cross-validation
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        X = self.scaler_X.transform(self.df[self.features].values)
        y = np.column_stack([self.scalers_y[t].transform(self.df[[t]]) for t in self.targets])

        cv_results = {t: [] for t in self.targets}

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
                
                rmse = np.sqrt(mean_squared_error(y_true_cv, y_pred_cv))
                cv_results[t].append(rmse)
        
        # Display
        for t in self.targets:
            rmses = cv_results[t]
            print(f"{t} : CV RMSE mean = {np.mean(rmses):.2f}, std = {np.std(rmses):.2f}")
        
        return cv_results

