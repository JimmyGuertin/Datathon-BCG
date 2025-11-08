# -*- coding: utf-8 -*-

import pandas as pd
import holidays
import numpy as np

# Trier par date et heure croissante
# -----------------------------
# 1. Trier par date et heure croissante
# -----------------------------
def order_by_date(df_champs):
    df_champs = df_champs.sort_values(by='Date et heure de comptage')
    df_champs = df_champs.reset_index(drop=True)
    return df_champs


# -----------------------------
# 2. Créer les features datetime et optionnellement compléter les heures manquantes
# -----------------------------
def create_datetime_features(df, fill_hours, datetime_col='Date et heure de comptage'):
    """
    Convertit une colonne en datetime, crée des features temporelles et peut compléter les heures manquantes.

    Parameters:
        df : pandas.DataFrame
        datetime_col : str, nom de la colonne datetime
        fill_hours : bool, si True complète toutes les heures manquantes entre début et fin
    Returns:
        df : pandas.DataFrame avec nouvelles colonnes et index complet si fill_hours=True
    """
    # Convertir en datetime (UTC pour homogénéité)
    df[datetime_col] = (pd.to_datetime(df[datetime_col], errors='coerce', utc=True).dt.tz_convert('Europe/Paris'))
    # Extraire features
    df['date'] = df[datetime_col].dt.date
    df['hour'] = df[datetime_col].dt.hour
    df['year'] = df[datetime_col].dt.year
    df['month'] = df[datetime_col].dt.month
    df['weekday'] = df[datetime_col].dt.weekday  # 0=lundi, 6=dimanche
    df['is_weekend'] = df['weekday'] >= 5

    # Optionnel : compléter toutes les heures
    if fill_hours:
        df = df.set_index(datetime_col)
        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')
        df = df.reindex(full_index)
        # Garder datetime comme colonne pour compatibilité
        df[datetime_col] = df.index

        # Recalculer features pour les nouvelles lignes (NaN pour les autres colonnes)
        df['date'] = df.index.date
        df['hour'] = df.index.hour
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['weekday'] = df.index.weekday
        df['is_weekend'] = df['weekday'] >= 5

    return df


def create_holidays(df_champs):
# Initialize France holidays
    fr_holidays = holidays.France(years=df_champs['date'].apply(lambda x: x.year).unique())

    # Add a new column 'is_holiday': True if the day is a French public holiday
    df_champs['is_holiday'] = df_champs['date'].apply(lambda x: x in fr_holidays)
    return(df_champs)


# Create a column describing the type of day
def day_type(row):
    if row['is_holiday']:
        return 'Public Holiday'
    else:
        return 'Normal Day'

def add_cyclic_features(df):
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

    return df


def fill_nan(df_champs):
    df_champs['Date et heure de comptage'] = pd.to_datetime(df_champs['Date et heure de comptage'], utc=True)

    # Trier chronologiquement
    df_champs = df_champs.sort_values('Date et heure de comptage')

    # Mettre la colonne de date comme index temporairement
    df_champs = df_champs.set_index('Date et heure de comptage')

    # Interpolation temporelle
    df_champs['Débit horaire'] = df_champs['Débit horaire'].interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
    df_champs['Taux d\'occupation'] = df_champs['Taux d\'occupation'].interpolate(method='time').fillna(method='bfill').fillna(method='ffill')

    return(df_champs)


def add_trafic_flow_around(df_champs, csv_path, on_cols=None, suffix="_around"):
    """
    Ajoute des features de trafic de tronçons aux alentours à partir d'un CSV externe.
    Les valeurs manquantes sont remplacées par celles de la colonne de base + un léger bruit.

    Paramètres
    ----------
    df_champs : pd.DataFrame
        Données principales (tronçons cibles, ex: Champs-Élysées).
    csv_path : str
        Chemin vers le fichier CSV contenant les données des tronçons voisins.
    on_cols : list of str, optional
        Liste des colonnes de jointure (par défaut ['Date et heure de comptage']).
    suffix : str, optional
        Suffixe ajouté aux colonnes du CSV fusionné pour éviter les collisions.

    Retourne
    --------
    pd.DataFrame
        DataFrame fusionné avec les nouvelles features de trafic.
    """

    datetime_col = "Date et heure de comptage"
    df_neighbors = pd.read_csv(csv_path,sep=";",encoding="utf-8")
    if "DÃ©bit horaire" in df_neighbors.columns:
        df_neighbors = df_neighbors.rename(columns={'DÃ©bit horaire':'Débit horaire'})
    df_neighbors = df_neighbors.loc[df_neighbors["Identifiant arc"]==4274,:]
    df_neighbors[datetime_col] = pd.to_datetime(df_neighbors[datetime_col], errors="coerce", utc=True)

    if on_cols is None:
        on_cols = [datetime_col]

    # Merge gauche
    df_merged = pd.merge(
        df_champs,
        df_neighbors[[datetime_col, "Débit horaire", "Taux d'occupation"]],
        how="left",
        on=datetime_col,
        suffixes=("", suffix)
    )

    # Vérif des valeurs manquantes avant remplacement
    missing_before = df_merged[[f"Débit horaire{suffix}", f"Taux d'occupation{suffix}"]].isna().sum().sum()
    print(f"[INFO] Valeurs manquantes après merge : {missing_before}")

    # Remplacement des NaN : valeur originale + léger bruit
    for col in ["Débit horaire", "Taux d'occupation"]:
        col_around = f"{col}{suffix}"

        # calcul du bruit : 1 à 2% du signal, bruit gaussien centré
        noise = np.random.normal(loc=0, scale=0.02 * df_merged[col].std(), size=len(df_merged))

        # remplacer les NaN
        df_merged[col_around] = df_merged[col_around].fillna(df_merged[col] + noise)

    # Vérif après remplissage
    missing_after = df_merged[[f"Débit horaire{suffix}", f"Taux d'occupation{suffix}"]].isna().sum().sum()
    print(f"[INFO] Valeurs manquantes après remplacement : {missing_after}")
    print(f"[INFO] Merge terminé : {df_merged.shape[0]} lignes, {df_merged.shape[1]} colonnes")

    return df_merged


def smooth_targets(df, targets, window=2):
    """
    Applique un lissage rolling mean sur les colonnes cibles sans changer leur nom.

    Parameters:
        df : pandas.DataFrame
        targets : list de str, colonnes à lisser
        window : int, taille de la fenêtre pour le rolling mean
    Returns:
        df_lisse : pandas.DataFrame avec les colonnes lissées
    """
    df_lisse = df.copy()
    for target in targets:
        df_lisse[target] = df_lisse[target].rolling(window=window, center=True, min_periods=1).mean()
        # Remplir les éventuels NaN par les valeurs originales
        df_lisse[target].fillna(df[target], inplace=True)
    return df_lisse

def vacances_by_zone(df):
    vacances = pd.read_csv('vacances/vacances.csv', parse_dates=['date'])

    # Renommer pour uniformité si nécessaire
    vacances = vacances.rename(columns={
        'vacances_zone_a': 'Vacances Zone A',
        'vacances_zone_b': 'Vacances Zone B',
        'vacances_zone_c': 'Vacances Zone C',
        'nom_vacances': 'Nom Vacances'
    })

    # Créer une colonne "Vacances Toutes Zones"
    vacances['Vacances Toutes Zones'] = vacances[['Vacances Zone A', 'Vacances Zone B', 'Vacances Zone C']].any(axis=1)

    # --- Fusionner avec ton DataFrame principal df ---

    vacances['date'] = pd.to_datetime(vacances['date']).dt.date

    df = df.merge(vacances, on='date',how='left')
    
    return(df)


def add_school_holidays_paris(df, date_col='Date et heure de comptage'):
    """
    Ajoute une colonne indicatrice 'Vacances Scolaires' pour Paris (zone C),
    incluant vacances hiver, printemps, été, Toussaint et Noël pour 2024 et 2025.
    """
    df[date_col] = pd.to_datetime(df[date_col])

    # Vacances scolaires pour Paris (zone C) sous forme de Series
    vacances_2024_2025 = pd.Series(pd.date_range('2024-10-19', '2024-11-04').tolist() +
                                   pd.date_range('2024-12-21', '2025-01-06').tolist() +
                                   pd.date_range('2025-02-15', '2025-03-03').tolist() +
                                   pd.date_range('2025-04-12', '2025-04-28').tolist() +
                                   pd.date_range('2025-07-05', '2025-09-01').tolist() +
                                   pd.date_range('2025-10-18', '2025-11-03').tolist() +
                                   pd.date_range('2025-12-20', '2026-01-05').tolist()
                                  )

    # Colonne indicatrice
    df['Vacances Scolaires Paris'] = df[date_col].dt.date.isin(vacances_2024_2025.dt.date).astype(int)

    return df


def merge_meteo(df_champs):
    # --- Lecture des fichiers météo ---
    df_meteo_1 = pd.read_csv("meteo/open-meteo-48.86N2.33E26m.csv", sep=",", header=2)
    df_meteo_2 = pd.read_csv("meteo/open-meteo-48.86N2.34E50m(1).csv", sep=",", header=2)
    df_meteo_3 = pd.read_csv("meteo/open-meteo-48.87N2.33E50m.csv", sep=",", header=2)

    # --- Harmonisation des colonnes ---
    # Si certaines colonnes n'existent pas dans tous les fichiers, on les ajoute
    all_columns = set(df_meteo_1.columns) | set(df_meteo_2.columns) | set(df_meteo_3.columns)
    for df in [df_meteo_1, df_meteo_2, df_meteo_3]:
        missing_cols = all_columns - set(df.columns)
        for col in missing_cols:
            df[col] = np.nan

    # --- Concaténation et nettoyage ---
    df_meteo = pd.concat([df_meteo_1, df_meteo_2, df_meteo_3], axis=0, ignore_index=True)
    df_meteo = df_meteo.drop_duplicates(subset=["time"])
    
    # On supprime la colonne précipitation_probability si elle existe
    if "precipitation_probability (%)" in df_meteo.columns:
        df_meteo = df_meteo.drop(columns=["precipitation_probability (%)"])

    # --- Conversion des dates ---
    df_champs["date"] = pd.to_datetime(df_champs["date"], errors="coerce")
    df_meteo["time"] = pd.to_datetime(df_meteo["time"], errors="coerce")

    # --- Fusion ---
    df_merged = df_champs.merge(df_meteo, left_on="date", right_on="time", how="left")

    return df_merged

def mark_outliers_and_special_events(df, targets, special_events_dict, top_n=20, iqr_factor=1.5):
    """
    Marque les outliers hauts et bas et les événements spéciaux.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes cibles et la colonne 'date'.
        targets (list of str): Colonnes cibles (ex: ['Débit horaire', "Taux d'occupation"]).
        special_events_dict (dict): Dictionnaire {date_string: 'event_name'}, ex: {'2025-02-02': 'course', ...}
        top_n (int): Nombre de valeurs les plus basses à considérer comme outliers bas.
        iqr_factor (float): Multiplicateur pour la détection IQR des outliers hauts.
        
    Returns:
        df (pd.DataFrame): DataFrame avec colonnes indicatrices pour outliers et événements spéciaux.
    """
    df = df.copy()
    
    # Convertir la colonne date en datetime si ce n'est pas déjà le cas
    if not np.issubdtype(df['date'].dtype, np.datetime64):
        df['date'] = pd.to_datetime(df['date'])
    
    for target in targets:
        col_high = f'{target}_outlier_high'
        col_low  = f'{target}_outlier_low'
        col_special = f'{target}_special_event'
        
        # Initialiser les colonnes
        df[col_high] = 0
        df[col_low] = 0
        df[col_special] = 0
        
        # --- Outliers bas et hauts automatiques ---
        # IQR
        Q1 = df[target].quantile(0.25)
        Q3 = df[target].quantile(0.75)
        IQR = Q3 - Q1
        high_threshold = Q3 + iqr_factor * IQR
        df[col_high] = (df[target] > high_threshold).astype(int)
        
        # Outliers bas : 20 plus bas
        low_indices = df[target].nsmallest(top_n).index
        df.loc[low_indices, col_low] = 1
        
        # --- Événements spéciaux ---
        for date_str, event_name in special_events_dict.items():
            event_date = pd.to_datetime(date_str)
            mask = df['date'].dt.date == event_date.date()
            df.loc[mask, col_special] = 1
    
    return df

def pipeline(df_champs,window,fill_hours,fillna=True):
    df_champs=order_by_date(df_champs)

    df_champs = create_datetime_features(df_champs,fill_hours)

    df_champs=vacances_by_zone(df_champs)

    df_champs = add_school_holidays_paris(df_champs)

    df_champs=create_holidays(df_champs)
    
    df_champs['day_type'] = df_champs.apply(day_type, axis=1)

    df_champs = add_cyclic_features(df_champs)
    if fillna:
        df_champs=fill_nan(df_champs)
    
    df_champs=merge_meteo(df_champs)
    
    targets = ['Débit horaire', "Taux d'occupation"]

    special_events_dict = {
        '2025-02-02': 'course',
        '2025-07-02': 'ceremonie',
        '2024-12-31': 'nouvel_an',
        '2024-11-11': 'armistice',
        '2025-07-14': 'fete_nationale'
    }

    df_champs = mark_outliers_and_special_events(df_champs, targets, special_events_dict, top_n=20)

    #df_champs = add_trafic_flow_around(df_champs,"data\comptages-routiers-permanents(5).csv")
    if window>0:
        df_champs = smooth_targets(df_champs, targets, window)


    return(df_champs)

def treat_nan_sts_peres(sts_peres_df):
    sts_peres_df = sts_peres_df.loc[sts_peres_df['Identifiant arc']==191,:]
    sts_peres_df['Date et heure de comptage'] = pd.to_datetime(sts_peres_df['Date et heure de comptage'], errors='coerce', utc=True)
    sts_peres_df = sts_peres_df.sort_values(by=['Date et heure de comptage'])
    cutoff = pd.Timestamp('2025-10-22 00:00:00', tz='Europe/Paris')

# Filtrer les lignes >= cutoff
    sts_peres_df_1 = sts_peres_df[
    sts_peres_df['Date et heure de comptage'] >= cutoff
].reset_index(drop=True)

    cutoff = pd.Timestamp('2024-11-12 04:00:00', tz='Europe/Paris')

# Filtrer les lignes <= cutoff
    sts_peres_df_2 = sts_peres_df[
    sts_peres_df['Date et heure de comptage'] <= cutoff
].reset_index(drop=True)

    sts_peres_df = pd.concat([sts_peres_df_1,sts_peres_df_2],axis=0)

    return sts_peres_df

def treat_nan_convention(convention_df):
    # Keep only data for the specific arc
    convention_df = convention_df.loc[convention_df['Identifiant arc'] == 5671, :]

    # Convert datetime column to timezone-aware timestamps
    convention_df['Date et heure de comptage'] = pd.to_datetime(
        convention_df['Date et heure de comptage'], errors='coerce', utc=True
    )

    # Sort chronologically
    convention_df = convention_df.sort_values(by=['Date et heure de comptage'])

    # Define date ranges (full days)
    start_1 = pd.Timestamp('2024-12-06T00:00:00+01:00')  # start of first day
    end_1 = pd.Timestamp('2025-02-11T23:59:59+01:00')    # end of last day
    cutoff_2 = pd.Timestamp('2025-05-21T00:00:00+01:00') # start of second period

    # Filter data for both periods (full days)
    df_period1 = convention_df[
        (convention_df['Date et heure de comptage'] >= start_1) &
        (convention_df['Date et heure de comptage'] <= end_1)
    ]
    df_period2 = convention_df[
        convention_df['Date et heure de comptage'] >= cutoff_2
    ]

    # Concatenate both periods
    convention_df = pd.concat([df_period1, df_period2]).reset_index(drop=True)

    return convention_df

def create_lag_features(df, targets, lags_hours, dropna=True):
    """
    Create lag features for given target columns and lag intervals.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the target columns.
    targets : list of str
        Columns for which lag features should be created.
    lags_hours : list of int
        List of lag intervals in hours (e.g. [1, 3, 24, 48, 168]).
    dropna : bool, default=True
        Whether to drop rows with NaN values generated by lagging.

    Returns
    -------
    df_lagged : pd.DataFrame
        DataFrame with added lag features (and optionally NaNs dropped).
    features : list of str
        List of names of the newly created lag feature columns.
    """

    df_lagged = df.copy()
    features = []

    # Create lag features
    for target in targets:
        for lag in lags_hours:
            col_name = f"{target}_lag_{lag}h"
            df_lagged[col_name] = df_lagged[target].shift(lag)
            features.append(col_name)

    # Optionally drop NaN rows generated by lagging
    if dropna:
        cols_to_check = [f"{t}_lag_{lag}h" for t in targets for lag in lags_hours] + targets
        df_lagged = df_lagged.dropna(subset=cols_to_check).copy()

    return df_lagged, features



import pandas as pd

# --- Sous-fonction 1 : création du squelette de test ---
def create_base_test_dataset():
    """Crée un DataFrame test (9–11 novembre 2025) avec toutes les features de base."""
    date_range = pd.date_range("2025-11-09 00:00:00", "2025-11-11 23:00:00", freq="H")

    df_test = pd.DataFrame({
        "Date et heure de comptage": date_range.strftime("%Y-%m-%dT%H:%M:%S+01:00"),
        "Débit horaire": [None] * len(date_range),
        "Taux d'occupation": [None] * len(date_range),
        "Etat trafic": [None] * len(date_range),
    })

    # Application du préprocessing
    df_test = order_by_date(df_test)
    df_test = create_datetime_features(df_test, fill_hours=True)
    df_test = vacances_by_zone(df_test)
    df_test = add_school_holidays_paris(df_test)
    df_test = create_holidays(df_test)
    df_test["day_type"] = df_test.apply(day_type, axis=1)
    df_test = add_cyclic_features(df_test)
    df_test = merge_meteo(df_test)

    # Ajout des colonnes d'outliers
    new_cols = [
        'Débit horaire_outlier_high',
        'Débit horaire_outlier_low',
        'Débit horaire_special_event',
        "Taux d'occupation_outlier_high",
        "Taux d'occupation_outlier_low",
        "Taux d'occupation_special_event",
    ]
    for col in new_cols:
        df_test[col] = False

    return df_test


# --- Sous-fonction 2 : copie des valeurs d'outliers depuis les données 2024 ---
def copy_outliers_from_2024(df_test, df_train, new_cols, name):
    """Copie les colonnes d’outliers/special_event de 2024 sur 2025 pour une période donnée."""
    df_train["date"] = pd.to_datetime(df_train["date"])

    mask = (
        (df_train["date"] >= pd.to_datetime("2024-11-09")) &
        (df_train["date"] <= pd.to_datetime("2024-11-11"))
    )
    df_period = df_train.loc[mask].copy()

    if len(df_period) != len(df_test):
        print(f"⚠️ Attention : {name} n’a pas le même nombre d’heures ({len(df_period)} vs {len(df_test)})")

    for col in new_cols:
        df_test[col] = df_period[col].reset_index(drop=True)

    
    return df_test


# --- Fonction principale ---
def create_test_dataset(champs_elysees_df, convention_df, sts_peres_df):
    """Crée les trois DataFrames test (Champs, Convention, Pères) avec copie des outliers 2024 et lags 72h/168h."""

    df_test = create_base_test_dataset()
    new_cols = [
        'Débit horaire_outlier_high',
        'Débit horaire_outlier_low',
        'Débit horaire_special_event',
        "Taux d'occupation_outlier_high",
        "Taux d'occupation_outlier_low",
        "Taux d'occupation_special_event",
    ]

    # Création des trois DataFrames
    df_test_champs_2025 = df_test.copy()
    df_test_convention_2025 = df_test.copy()
    df_test_peres_2025 = df_test.copy()

    # --- Copie des valeurs d’outliers depuis 2024 ---
    df_test_champs_2025 = copy_outliers_from_2024(df_test_champs_2025, champs_elysees_df, new_cols, "Champs-Élysées")
    df_test_convention_2025 = copy_outliers_from_2024(df_test_convention_2025, sts_peres_df, new_cols, "Convention")
    df_test_peres_2025 = copy_outliers_from_2024(df_test_peres_2025, sts_peres_df, new_cols, "Pères")

    lag_hours = [72,168]
    targets = ["Débit horaire", "Taux d'occupation"]
    for df_test, df_train, name in [
        (df_test_champs_2025, champs_elysees_df, "Champs-Élysées"),
        (df_test_convention_2025, convention_df, "Convention"),
        (df_test_peres_2025, sts_peres_df, "Pères")
    ]:
        # S'assurer que date est bien datetime
        df_train["date"] = pd.to_datetime(df_train["date"])

        # Créer un index datetime pour le train à partir de date+hour
        df_train_indexed = df_train.copy()
        df_train_indexed["datetime_index"] = df_train_indexed["date"] + pd.to_timedelta(df_train_indexed["hour"], unit="h")
        df_train_indexed = df_train_indexed.set_index("datetime_index")

        for target in targets:
            for lag in lag_hours:
                col_name = f"{target}_lag_{lag}h"
                lag_values = []

                for idx, row in df_test.iterrows():
                    dt = pd.to_datetime(row["date"]) + pd.Timedelta(hours=row["hour"])
                    # --- 👇 Ajustement du lag selon la date ---
                    if dt.date() == pd.to_datetime("2025-11-11").date() and lag == 72:
                        lag_to_use = 96  # on force 96h le 11 novembre
                    else:
                        lag_to_use = lag  # sinon, on garde 72h
                    dt_lag = dt - pd.Timedelta(hours=lag_to_use)

                    if dt_lag in df_train_indexed.index:
                        lag_values.append(df_train_indexed.loc[dt_lag, target])
                    else:
                        lag_values.append(np.nan)

                df_test[col_name] = lag_values
    return df_test_champs_2025, df_test_convention_2025, df_test_peres_2025
