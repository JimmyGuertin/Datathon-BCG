import pandas as pd
import numpy as np

class Preprocessor:
    def __init__(self, df):
        self.df = df
    
    def create_datetime_features(self, df: pd.DataFrame, holidays_df: pd.DataFrame, datetime_col: str ='Date et heure de comptage') -> pd.DataFrame:
        """
        Converts a column to datetime and creates additional columns:
            - date: date without timestamp
            - hour: time
            - year: year
            - month: month
            - weekday: day of the week (0=Monday, 6=Sunday)
            - is_weekend: True if Saturday or Sunday
            - is_holiday: Binary public and school holidays
            - hour_sin, hour_cos: cyclic encoding (daily seasonality)
            - weekday_sin, weekday_cos: cyclic encoding (weekly seasonality)
            - month_sin, month_cos: cyclic encoding (weekly seasonality)


            Parameters:
                df: pandas.DataFrame, street dataset
                datetime_col: str, name of the datetime column
                holidays_df: pandas.DataFrame, French public holidays dataset
            Returns:
                df: pandas.DataFrame with new columns

        """

        # Convert to datetime (not in UTC to keep winter and summer french time)
        df['Date et heure de comptage'] = pd.to_datetime(df['Date et heure de comptage'], errors='coerce', utc=True)

        # Convert to tz-naive (Paris local time)
        df['Date et heure de comptage'] = df['Date et heure de comptage'].dt.tz_convert('Europe/Paris').dt.tz_localize(None)


        # Extract features
        df['date'] = df[datetime_col].dt.date
        df['hour'] = df[datetime_col].dt.hour
        df['year'] = df[datetime_col].dt.year
        df['month'] = df[datetime_col].dt.month
        df['weekday'] = df[datetime_col].dt.weekday  # 0=lundi, 6=dimanche
        df['is_weekend'] = df['weekday'] >= 5

        # Add cyclic features
        self.add_cyclic_features(df)

        # Add holidays
        self.add_holidays(df, holidays_df)

        return df

    def add_cyclic_features(self, df):
        """
        Ajoute des colonnes sin/cos pour les features cycliques : heure, jour de semaine, mois.
        """

        # Heure (0-23)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Jour de la semaine (0-6)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)

        # Mois (1-12)
        df['month_sin'] = np.sin(2 * np.pi * (df['month']-1) / 12)
        df['month_cos'] = np.cos(2 * np.pi * (df['month']-1) / 12)

        # Optionnel : jour de l'année (1-365/366)
        df['day_of_year'] = df['Date et heure de comptage'].dt.dayofyear
        df['dayofyear_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    def add_holidays(self, df: pd.DataFrame, holidays_df: pd.DataFrame):
        """
        Adds columns indicating whether a date is within French school holidays.

        Parameters:
            df (pd.DataFrame): main dataset
            holidays_df (pd.DataFrame): vacation periods dataset
            datetime_col (str): datetime column in df

        Returns:
            pd.DataFrame: updated df with 'is_holiday'
        """
        df['is_holiday'] = False
        
        # Convert to datetime not in UTC
        holidays_df['Date de début'] = pd.to_datetime(holidays_df['Date de début'], utc=False, errors='coerce').dt.date
        holidays_df['Date de fin'] = pd.to_datetime(holidays_df['Date de fin'], utc=False, errors='coerce').dt.date

        # Only keep Zone C (Paris)
        holidays_df = holidays_df[holidays_df['Zones'] == 'Zone C']

        for _, row in holidays_df.iterrows():
            mask = (df["date"] >= row['Date de début']) & (df["date"] <= row['Date de fin'])
            df.loc[mask, 'is_holiday'] = True
        
    def fill_nan(self, df: pd.DataFrame):
        df['Date et heure de comptage'] = pd.to_datetime(df['Date et heure de comptage'], utc=False)

        # Trier chronologiquement
        df = df.sort_values('Date et heure de comptage')

        # Mettre la colonne de date comme index temporairement
        df = df.set_index('Date et heure de comptage')

        # Interpolation temporelle
        df['Débit horaire'] = df['Débit horaire'].interpolate(method='time')
        df['Taux d\'occupation'] = df['Taux d\'occupation'].interpolate(method='time')
        return df
    
    def add_weather(self, df: pd.DataFrame, weather_df: pd.DataFrame, datetime_col: str = 'Date et heure de comptage'):
        """
        Adds weather features to the traffic dataframe by merging on datetime.

        Parameters:
            df (pd.DataFrame): traffic dataset
            weather_df (pd.DataFrame): weather dataset, must have 'time' column
            datetime_col (str): datetime column in traffic df

        Notes:
            - Assumes weather_df['time'] and df[datetime_col] are compatible datetimes
            - Merge is done on datetime rounded to hour
        """

        # Convert weather time to datetime (if not already)
        weather_df['time'] = pd.to_datetime(weather_df['time'], errors='coerce', utc=False)

        # Ensure traffic datetime is also datetime
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce', utc=False)

        # Optional: round both to hour for exact matching
        df['hour_time'] = df[datetime_col].dt.floor('H')
        weather_df['hour_time'] = weather_df['time'].dt.floor('H')

        # Merge on the floored hour
        df = df.merge(weather_df, on='hour_time', how='left', suffixes=('', '_weather'))

        # Drop helper column if you want
        df.drop(columns=['hour_time'], inplace=True)

        return df
    
    def preprocess_all(self, holidays_df: pd.DataFrame, weather_df: pd.DataFrame, datetime_col: str='Date et heure de comptage'):
        """
        Runs all preprocessing steps:
        - Fill NaN
        - Create datetime features
        - Add weather data
        """
        # Create datetime features and holidays
        self.create_datetime_features(self.df, holidays_df)

        # Merge weather data
        self.df = self.add_weather(self.df, weather_df, datetime_col=datetime_col)

        # Fill missing traffic values
        self.df = self.fill_nan(self.df)
        return self.df    