"""Tous les paramètres du batch se règlent ici. Les taux sont en décimal."""
from itertools import product

PROFILS = {
    'plate': (0.025, 0.025),
    'montante': (0.015, 0.040),
    'descendante': (0.040, 0.015),
}
# Chaque paire décrit le taux zéro court et sa limite à long terme.
ECHELLE_COURBE = 10.0  # années : z(T) = long + (court - long) * exp(-T / échelle)

VITESSES_RETOUR = [0.05, 0.10, 0.30]
VOLATILITES = [0.003, 0.005, 0.010]
MATURITES = [15]
SEEDS = list(range(1, 51))

PARAMETRES = {
    'debut': '202501',
    'nominal': 100_000.0,
    'chemins_conditionnels': 300,
    'marge_commerciale': 0.002,
    'marge_liquidite': 0.005,
    'marge_refinancement': 0.004,
    'indemnite_prepaiement': 0.0,
}

from dotenv import load_dotenv
import os

load_dotenv()

PREPAIEMENT = {
    'intercept': float(os.getenv("PREPAY_INTERCEPT")),
    'beta_change': float(os.getenv("PREPAY_BETA_CHANGE")),
    'beta_gain': float(os.getenv("PREPAY_BETA_GAIN")),
    'minimum_rate': float(os.getenv("MINIMUM_RATE")),
    'threshold': 0.50,
    'almin': float(os.getenv("ALMIN")),
}

# False : aucun détail ; 'transactions' : swaps pris ; True : prix à chaque mois.
SAUVER_SWAPS = 'transactions'

PRODUITS = ['fixe', 'variable', 'protected']
VARIABLE = {
    'revision_mois': 36,
    'hausse_max': 0.0,         # La référence protected ne dépasse pas son niveau initial.
    'vol_normale': None,      # None : approximation normale déduite de a et sigma HW.
}


def experiences():
    for profil, a, sigma, maturite, seed in product(
        PROFILS, VITESSES_RETOUR, VOLATILITES, MATURITES, SEEDS
    ):
        yield {
            'profil': profil,
            'a': a,
            'sigma': sigma,
            'maturite_ans': maturite,
            'seed': seed,
        }
