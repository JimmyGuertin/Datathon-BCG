#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scrappe les matchs d'une page d'événements sportifs sur paris.fr
et écrit un CSV propre sur la sortie standard.

Exemples d'utilisation :

    # PSG
    python script.py \
      "https://www.paris.fr/evenements/paris-saint-germain-le-calendrier-foot-2025-2026-des-matchs-a-domicile-90939" \
      > psg_2025_2026.csv

    # Stade Français
    python script.py \
      "https://www.paris.fr/evenements/stade-francais-paris-le-calendrier-complet-des-matchs-a-domicile-saison-2025-2026-90960" \
      > stade_francais_2025_2026.csv
"""

import argparse
import csv
import sys

import requests
from bs4 import BeautifulSoup


def scrape_paris_sport_events(url, section_title_contains_list=None):
    """
    Scrappe les matchs listés dans la section calendrier d'une page paris.fr.

    :param url: URL de la page (paris.fr/evenements/...)
    :param section_title_contains_list: liste de fragments de texte à chercher
                                        dans les titres de section (h2/h3)
    :return: liste de dicts avec les champs :
             competition, date, time, opponent, source_url
    """
    if section_title_contains_list is None:
        section_title_contains_list = [
            "dates et horaires",                 # ex. PSG
            "programme complet des matchs",      # ex. Stade Français
            "calendrier des matchs",             # fallback général
            "programme complet des rencontres",
        ]

    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # 1) Trouver le DERNIER <h2>/<h3> qui matche (et pas le premier)
    section_h = None
    for h in soup.find_all(["h2", "h3"]):
        text = h.get_text(strip=True).lower()
        if any(fragment.lower() in text for fragment in section_title_contains_list):
            section_h = h  # on ne fait PAS de break -> on garde le dernier

    if section_h is None:
        raise RuntimeError(
            f"Aucune section calendrier trouvée sur {url} "
            f"(titres cherchés : {section_title_contains_list})"
        )

    events = []
    current_competition = None

    # 2) Parcourir les éléments après ce titre
    for sibling in section_h.find_all_next():
        # Option : arrêter quand on atteint un vrai bloc de fin ("À lire aussi", "Sélections")
        if sibling.name == "h2":
            h2_text = sibling.get_text(strip=True).lower()
            if "à lire aussi" in h2_text or "sélections" in h2_text:
                break

        # Un <h3> = nom de compétition (Ligue 1, Top 14, etc.)
        if sibling.name == "h3":
            current_competition = sibling.get_text(strip=True)

        # Un <li> = une ligne "date - heure : adversaire"
        if sibling.name == "li":
            raw_text = sibling.get_text(strip=True)

            date_str = None
            time_str = None
            opponent = None

            # Séparer gauche (date/heure) et droite (adversaire)
            left = raw_text
            if ":" in raw_text:
                left, opponent = raw_text.split(":", 1)
                opponent = opponent.strip().rstrip(".")

            # Dans la partie gauche, séparer date et heure avec " - " si présent
            if "-" in left:
                date_part, time_part = left.split("-", 1)
                date_str = date_part.strip()
                time_str = time_part.strip()
            else:
                date_str = left.strip()

            events.append(
                {
                    "competition": current_competition or "",
                    "date": date_str or "",
                    "time": time_str or "",
                    "opponent": opponent or "",
                    "source_url": url,
                }
            )

    return events


def main():
    parser = argparse.ArgumentParser(
        description="Scraper de calendrier sportif depuis paris.fr vers CSV."
    )
    parser.add_argument(
        "url",
        help="URL de la page d'événement paris.fr (calendrier des matchs à domicile, etc.)",
    )
    parser.add_argument(
        "-s",
        "--section",
        action="append",
        dest="sections",
        default=None,
        help=(
            "Fragment de texte pour identifier le titre de la section calendrier "
            "(peut être répété). Si non fourni, une liste par défaut est utilisée."
        ),
    )

    args = parser.parse_args()

    events = scrape_paris_sport_events(
        args.url,
        section_title_contains_list=args.sections,
    )

    writer = csv.writer(sys.stdout, delimiter=";", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["competition", "date", "time", "opponent", "source_url"])

    for e in events:
        writer.writerow(
            [
                e["competition"],
                e["date"],
                e["time"],
                e["opponent"],
                e["source_url"],
            ]
        )


if __name__ == "__main__":
    main()
