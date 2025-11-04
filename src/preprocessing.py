import pandas as pd
import numpy as np
from typing import Optional, List, Dict

class Preprocessor:
    def __init__(self, df):
        self.df = df
        self.targets =  ['Débit horaire', "Taux d'occupation"]
    
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
            df.loc[mask, 'holiday_name'] = row['Description']
        
    def flag_outliers_iqr(
        self,
        df: pd.DataFrame,
        columns: Optional[list] = None,
        multiplier: float = 1.5,
        suffix: str = '_outlier_iqr',
        record_bounds: bool = True,
    ) -> pd.DataFrame:
        """
        Add columns to flag outliers when the given variables expand outside its boundaries
        """
        # Déterminer les colonnes à traiter
        if columns is None:
            cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        else:
            cols = [c for c in columns if c in df.columns]

        bounds: Dict[str, tuple] = {}
        for c in cols:
            series = df[c]
            # Coercion prudente en numérique si nécessaire
            if not pd.api.types.is_numeric_dtype(series):
                series = pd.to_numeric(series, errors='coerce')

            if series.dropna().empty:
                lower = np.nan
                upper = np.nan
            else:
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - multiplier * iqr
                upper = q3 + multiplier * iqr

            bounds[c] = (lower, upper)
            flag_col = f"{c}{suffix}"
            if pd.isna(lower) or pd.isna(upper):
                df[flag_col] = False
            else:
                mask = (series < lower) | (series > upper)
                df[flag_col] = mask.fillna(False)

        if record_bounds:
            self._iqr_bounds = bounds

        return df

    def flag_outliers_on_targets(
        self,
        df: pd.DataFrame,
        targets: Optional[List[str]] = None,
        multiplier: float = 1.5,
        suffix: str = '_outlier_iqr',
        record_bounds: bool = True,
    ) -> pd.DataFrame:
        """
        Convenience wrapper to flag outliers only on the forecasting target columns.

        By default, if `targets` is None, uses the common targets
        ['Débit horaire', "Taux d'occupation"]. Only the targets present in `df`
        will be processed.
        """
        if targets is None:
            targets = ['Débit horaire', "Taux d'occupation"]

        present_targets = [t for t in targets if t in df.columns]
        if not present_targets:
            # nothing to do
            return df

        return self.flag_outliers_iqr(
            df, columns=present_targets, multiplier=multiplier, suffix=suffix, record_bounds=record_bounds
        )
        
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

    def add_sports_events(self, df: pd.DataFrame, sports_df: pd.DataFrame, datetime_col: str = 'Date et heure de comptage'):
        """
        Adds sports event features to the traffic dataframe by merging on datetime.

        Parameters:
            df (pd.DataFrame): traffic dataset
            sports_df (pd.DataFrame): sports events dataset, must have 'event_time' column
            datetime_col (str): datetime column in traffic df

        Notes:
            - Assumes sports_df['event_time'] and df[datetime_col] are compatible datetimes
            - Merge is done on datetime rounded to hour
        """

        # Convert sports event time to datetime (if not already)
        sports_df['date_utc'] = pd.to_datetime(sports_df['date_utc'], errors='coerce', utc=True)
        sports_df['date_paris'] = sports_df['date_utc'].dt.tz_convert('Europe/Paris').dt.tz_localize(None)

        # Round to hour for matching
        sports_df['hour_time'] = sports_df['date_paris'].dt.floor('H')

        # List of French teams and big teams
        french_teams = [
            'Paris Saint-Germain', 'Olympique de Marseille', 'Olympique Lyonnais', 
            'AS Monaco', 'Lille OSC', 'RC Lens', 'AS Saint-Étienne', 
            'Equipe de France', 'France'
        ]

        big_teams = [
            'Real Madrid', 'Barcelona', 'FC Barcelona', 'Manchester City', 
            'Manchester United', 'Liverpool', 'Bayern Munich', 'Chelsea',
            'Arsenal', 'Inter', 'AC Milan', 'Juventus'
        ]

        major_competitions = [
            'UEFA Champions League', 'UEFA Euro', 'World Cup', 'Coupe du Monde',
            'Ligue 1', 'Europa League', 'Euro', 'Olympic Games'
        ]

        # Filter for relevant events
        mask = (
            sports_df['home_team'].isin(french_teams)
            | sports_df['away_team'].isin(french_teams)
            | sports_df['home_team'].isin(big_teams)
            | sports_df['away_team'].isin(big_teams)
            | sports_df['competition_name'].isin(major_competitions)
        )

        sports_filtered = sports_df[mask].copy()

        # Create a summary of events per date
        sports_filtered['sport_event_name'] = (
            sports_filtered['competition_name'] + ' : ' +
            sports_filtered['home_team'] + ' vs ' + sports_filtered['away_team']
        )

        event_summary = (
            sports_filtered.groupby('hour_time')['sport_event_name']
            .apply(lambda x: ', '.join(sorted(set(x))))
            .reset_index()
        )

        # Optional: round both to hour for exact matching
        df['hour_time'] = df['Date et heure de comptage'].dt.floor('H')
        df = df.merge(event_summary, on='hour_time', how='left')

        # Create binary indicator
        df['is_sport_event'] = df['sport_event_name'].notna()

        # Drop helper column if you want
        df.drop(columns=['hour_time'], inplace=True)

        return df
    
    def fit_transform(
        self, holidays_df: pd.DataFrame, weather_df: pd.DataFrame, sports_df: pd.DataFrame,
        datetime_col: str='Date et heure de comptage'
    ):
        """
        Runs all preprocessing steps:
        - Fill NaN
        - Create datetime features
        - Add weather data
        - Add holidays and sport events
        """
        # Create datetime features and holidays
        self.create_datetime_features(self.df, holidays_df)

        # Merge weather data
        self.df = self.add_weather(self.df, weather_df, datetime_col=datetime_col)

        # Add sport events
        self.df = self.add_sports_events(self.df, sports_df)

        # Fill missing traffic values
        self.df = self.fill_nan(self.df)

        # Add outliers flags
        self.df = self.flag_outliers_on_targets(df=self.df, targets=self.targets, record_bounds=False)

        return self.df
    