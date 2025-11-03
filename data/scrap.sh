#!/usr/bin/env bash

'''
Fichier : scrap.sh
Description : Script bash pour exécuter le script de scraping Python et enregistrer les résultats dans un fichier CSV.
Ce script fonctionne pour les sites d événements sportifs de la ville de Paris
'''

# Choisir le nom du fichier csv généré
NAME="match_parisfc"

# Détermine le dossier où se trouve ce script (et donc scapping_sport.py)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="/usr/local/bin/python3"
PY_SCRIPT="${SCRIPT_DIR}/scapping_sport.py"

URL="https://www.paris.fr/evenements/matchs-du-paris-fc-pour-la-saison-2025-2026-le-calendrier-complet-90428"
OUTPUT_CSV="${SCRIPT_DIR}/${NAME}.csv"

# Exécution du script Python avec redirection vers le CSV
"$PYTHON_BIN" "$PY_SCRIPT" "$URL" > "$OUTPUT_CSV"
