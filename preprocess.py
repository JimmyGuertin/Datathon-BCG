import pandas as pd
import holidays
import numpy as np

# Trier par date et heure croissante
def order_by_date(df_champs):
    df_champs = df_champs.sort_values(by='Date et heure de comptage')

# Si tu veux réinitialiser les index après le tri
    df_champs = df_champs.reset_index(drop=True)
    
    return(df_champs)

def create_datetime_features(df, datetime_col='Date et heure de comptage'):
    """
    Convertit une colonne en datetime et crée des colonnes supplémentaires :
    - day : date sans heure
    - hour : heure
    - year : année
    - month : mois
    - weekday : jour de la semaine (0=lundi, 6=dimanche)
    - is_
    end : True si samedi ou dimanche

    Parameters:
        df : pandas.DataFrame
        datetime_col : str, nom de la colonne datetime
    Returns:
        df : pandas.DataFrame avec nouvelles colonnes
    """

    # Convertir en datetime (UTC pour homogénéité)
    df[datetime_col] = (pd.to_datetime(df[datetime_col], errors='coerce', utc=True))#.dt.tz_convert('Europe/Paris'))
    # Extraire features
    df['date'] = df[datetime_col].dt.date
    df['hour'] = df[datetime_col].dt.hour
    df['year'] = df[datetime_col].dt.year
    df['month'] = df[datetime_col].dt.month
    df['weekday'] = df[datetime_col].dt.weekday  # 0=lundi, 6=dimanche
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
    df_champs['Débit horaire'] = df_champs['Débit horaire'].interpolate(method='time')
    df_champs['Taux d\'occupation'] = df_champs['Taux d\'occupation'].interpolate(method='time')
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



