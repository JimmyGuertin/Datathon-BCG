import pandas as pd
import numpy as np

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
    df_meteo_1 = pd.read_csv("meteo/open-meteo-48.86N2.34E50m(1).csv",sep=",",header=2)
    df_meteo_2 = pd.read_csv("meteo/open-meteo-48.87N2.33E50m.csv",sep=",",header=2)
    df_meteo = pd.concat([df_meteo_1,df_meteo_2],axis=0)
    df_meteo = df_meteo.drop_duplicates()
    df_meteo = df_meteo.drop(columns=['precipitation_probability (%)'])

    df_champs['date']=pd.to_datetime(df_champs['date'])

    df_meteo["time"] = pd.to_datetime(df_meteo["time"])  
    
    df_champs=df_champs.merge(df_meteo,right_on='time',left_on='date',how='left')
    return(df_champs)


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