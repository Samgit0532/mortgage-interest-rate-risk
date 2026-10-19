"""Hull–White à paramètres fixes et courbes initiales synthétiques."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from config import PROFILS, ECHELLE_COURBE

@dataclass
class Settings:
    nombre_chemins: int = 300
    graine: int = 1
    hw_a: float = .1
    hw_sigma: float = .005
    hw_mode: str = "fixe"


def generer_simulation(depot, debut, fin):
    """Un central réalisé et des futurs conditionnels à chaque fin de mois."""
    settings = depot.settings
    if settings.hw_a <= 0 or settings.hw_sigma < 0:
        raise ValueError('HW : a doit être positif et sigma positif ou nul.')
    if not np.isfinite([settings.hw_a, settings.hw_sigma]).all():
        raise ValueError('Les paramètres HW doivent être finis.')

    curve = depot.courbe(debut)
    parameters = HWParameters(settings.hw_a, settings.hw_sigma)
    dates = grille_mensuelle(curve.date, fin)
    seeds = np.random.SeedSequence(settings.graine).spawn(len(dates) + 1)
    central = generer_chemins(curve, fin, parameters, 1, np.random.default_rng(seeds[0]))
    observed = pd.DataFrame({'taux_6m': central.taux_6m[0]}, index=central.dates).loc[dates]

    conditional = {}
    for i, date in enumerate(dates[:-1]):
        current = depot.courbe_conditionnelle(date, central, parameters)
        conditional[date.strftime('%Y%m')] = generer_chemins(
            current, fin, parameters, settings.nombre_chemins,
            np.random.default_rng(seeds[i + 1]),
        )

    fixed = pd.DataFrame({'a': settings.hw_a, 'sigma': settings.hw_sigma}, index=dates[:-1])
    return Simulation(observed, conditional, fixed, central)


@dataclass
class Curve:
    """Courbe de taux zéro, exprimée en taux continus et temps ACT/365."""
    date: pd.Timestamp
    temps: np.ndarray
    taux_zero: np.ndarray

    def __post_init__(self):
        self.date = pd.Timestamp(self.date).normalize()
        self.temps = np.asarray(self.temps, dtype=float)
        self.taux_zero = np.asarray(self.taux_zero, dtype=float)
        if (
            self.temps.ndim != 1
            or len(self.temps) < 2
            or self.temps.shape != self.taux_zero.shape
            or not np.all(np.isfinite(self.temps))
            or not np.all(np.isfinite(self.taux_zero))
            or self.temps[0] != 0
            or np.any(np.diff(self.temps) <= 0)
        ):
            raise ValueError('Piliers de courbe invalides : commencer à zéro et croître strictement.')

    def zero(self, t):
        """Taux zéro interpolé. L'extrapolation est refusée."""
        t = np.asarray(t, dtype=float)
        if np.any(t < 0) or np.any(t > self.temps[-1]):
            raise ValueError('La maturité demandée dépasse la courbe.')
        return np.interp(t, self.temps, self.taux_zero)

    def discount(self, t):
        """Prix aujourd'hui d'un zéro-coupon : exp(-taux zéro × durée)."""
        t = np.asarray(t, dtype=float)
        return np.exp(-self.zero(t) * t)

    def forward_instantane(self, t):
        """Taux forward instantané calculé à partir de la courbe zéro."""
        t = np.asarray(t, dtype=float)
        i = np.clip(np.searchsorted(self.temps, t, side='right') - 1, 0, len(self.temps) - 2)
        pente = np.diff(self.taux_zero) / np.diff(self.temps)
        return self.zero(t) + t * pente[i]

    def ancrer_sur_6m(self, date, taux_6m):
        """Décale la courbe pour retrouver le taux 6M observé à cette date."""
        date = pd.Timestamp(date).normalize()
        fin_6m = date + pd.DateOffset(months=6)
        duree = (fin_6m - date).days / 365
        accrual_360 = (fin_6m - date).days / 360
        if 1 + taux_6m * accrual_360 <= 0:
            raise ValueError('Taux 6M incompatible avec un facteur positif.')
        taux_zero_vise = np.log1p(taux_6m * accrual_360) / duree
        decalage = taux_zero_vise - float(self.zero(duree))
        return Curve(date, self.temps.copy(), self.taux_zero + decalage)

@dataclass
class HWParameters:
    a: float
    sigma: float

@dataclass
class Paths:
    """Matrices de taux et d'actualisation pour tous les chemins d'une date."""
    dates: pd.DatetimeIndex
    taux_6m: np.ndarray
    facteurs_actualisation: np.ndarray
    courbe: Curve | None = None
    parameters: HWParameters | None = None
    taux_court: np.ndarray | None = None

    def __post_init__(self):
        self.dates = pd.DatetimeIndex(self.dates).normalize()
        self.taux_6m = np.asarray(self.taux_6m, dtype=float)
        self.facteurs_actualisation = np.asarray(self.facteurs_actualisation, dtype=float)
        if (
            len(self.dates) == 0
            or self.dates.has_duplicates
            or not self.dates.is_monotonic_increasing
            or self.taux_6m.ndim != 2
            or self.taux_6m.shape[0] == 0
            or self.taux_6m.shape[1] != len(self.dates)
            or self.facteurs_actualisation.shape != self.taux_6m.shape
            or not np.all(np.isfinite(self.taux_6m))
            or not np.all(np.isfinite(self.facteurs_actualisation))
            or np.any(self.facteurs_actualisation <= 0)
        ):
            raise ValueError('Chemins : dates, taux ou actualisation invalides.')
        if self.taux_court is not None and np.asarray(self.taux_court).shape != self.taux_6m.shape:
            raise ValueError('États de taux court incompatibles avec les chemins.')

@dataclass
class Simulation:
    scenario_central: pd.DataFrame
    scenarios_conditionnels: dict
    parametres: pd.DataFrame
    central_paths: Paths | None = None

def grille_mensuelle(debut, fin):
    """Construit les dates mensuelles de simulation jusqu'à la maturité."""
    debut = pd.Timestamp(debut).normalize()
    fin = pd.Timestamp(fin).normalize()
    dates = pd.date_range(debut, fin, freq=pd.offsets.MonthEnd())
    dates = [debut] + [date for date in dates if date > debut]
    if dates[-1] != fin:
        dates.append(fin)
    return pd.DatetimeIndex(dates)

def grille_pricing(debut, fin):
    """Grille mensuelle enrichie aux dates de fixing des swaps à départ différé.

    Deuxième jour ouvré du mois et deux jours ouvrés avant chaque fin de mois.
    Le calendrier ouvré du prototype exclut les weekends (jours fériés non inclus).
    """
    monthly = grille_mensuelle(debut, fin)
    points = set(monthly)
    months = pd.period_range(
        pd.Timestamp(debut).to_period('M'),
        pd.Timestamp(fin).to_period('M'),
        freq='M'
    )
    for month in months:
        prior_end = month.to_timestamp() - pd.Timedelta(days=1)
        month_end = month.to_timestamp(how='end').normalize()
        for date in (prior_end + pd.offsets.BDay(2), month_end - pd.offsets.BDay(2)):
            if pd.Timestamp(debut) < date < pd.Timestamp(fin):
                points.add(pd.Timestamp(date).normalize())
    return pd.DatetimeIndex(sorted(points))

def generer_chemins(courbe, fin, parameters, nombre, generateur):
    """Génère les taux 6M et les facteurs d'actualisation des chemins."""
    if nombre < 1:
        raise ValueError('Le nombre de chemins doit être positif.')
    dates = grille_pricing(courbe.date, fin)
    temps = (dates - dates[0]).days.to_numpy() / 365
    a, sigma = (parameters.a, parameters.sigma)
    if not np.isfinite(a) or a <= 0 or (not np.isfinite(sigma)) or (sigma < 0):
        raise ValueError('HWParameters Hull–White invalide.')
    x = np.zeros((nombre, len(dates)))
    integrale_taux = np.zeros_like(x)
    noeuds, poids = np.polynomial.legendre.leggauss(12)

    def B(duree):
        return -np.expm1(-a * duree) / a

    def phi(t):
        return courbe.forward_instantane(t) + 0.5 * sigma ** 2 * B(t) ** 2
    for k in range(1, len(dates)):
        dt = temps[k] - temps[k - 1]
        variance_x = sigma ** 2 * (1 - np.exp(-2 * a * dt)) / (2 * a)
        normale_x = generateur.standard_normal(nombre)
        x[:, k] = np.exp(-a * dt) * x[:, k - 1] + np.sqrt(variance_x) * normale_x
        u = dt * (noeuds + 1) / 2
        w = dt * poids / 2
        bu = B(u)
        decay = np.exp(-a * u)
        var_x = sigma ** 2 * np.sum(w * decay ** 2)
        covariance = sigma ** 2 * np.sum(w * decay * bu)
        var_integrale = sigma ** 2 * np.sum(w * bu ** 2)
        partage = covariance / np.sqrt(var_x) if var_x > 0 else 0
        reste = np.sqrt(max(var_integrale - partage ** 2, 0))
        normale_integrale = generateur.standard_normal(nombre)
        alea_integrale = partage * normale_x + reste * normale_integrale
        u_absolu = temps[k - 1] + u
        correction_deterministe = np.log(courbe.discount(temps[k - 1])) - np.log(courbe.discount(temps[k])) + 0.5 * sigma ** 2 * np.sum(w * B(u_absolu) ** 2)
        integrale_taux[:, k] = integrale_taux[:, k - 1] + B(dt) * x[:, k - 1] + correction_deterministe + alea_integrale
    taux_court = x + phi(temps)[None, :]
    taux_6m = np.zeros_like(taux_court)
    for k, date in enumerate(dates):
        date_6m = date + pd.DateOffset(months=6)
        duree_6m = (date_6m - date).days / 365
        maturite = temps[k] + duree_6m
        b = B(duree_6m)
        prix_oblig = courbe.discount(maturite) / courbe.discount(temps[k]) * np.exp(
            -b * (taux_court[:, k] - courbe.forward_instantane(temps[k])) - sigma ** 2 / (4 * a) * (1 - np.exp(-2 * a * temps[k])) * b ** 2
        )
        accrual_360 = (date_6m - date).days / 360
        taux_6m[:, k] = (1 / prix_oblig - 1) / accrual_360
    facteurs = np.exp(-integrale_taux)
    return Paths(dates, taux_6m, facteurs, courbe, parameters, taux_court)

def parallel_bump(paths: Paths, shock: float) -> Paths:
    """Même choc zéro continu pour discounting et projections 6M.

    Les fixings historiques sont stockés dans les swaps et ne sont pas choqués.
    """
    dates = pd.DatetimeIndex(paths.dates)
    elapsed = (dates - dates[0]).days.to_numpy(dtype=float) / 365
    days_6m = np.array([(d + pd.DateOffset(months=6) - d).days for d in dates])
    delta_360, delta_365 = (days_6m / 360, days_6m / 365)
    rates = np.asarray(paths.taux_6m, dtype=float)
    factors = np.asarray(paths.facteurs_actualisation, dtype=float)
    bumped_rates = (
        (1 + rates * delta_360[None, :]) * np.exp(shock * delta_365[None, :]) - 1
    ) / delta_360[None, :]
    bumped_curve = None if paths.courbe is None else ParallelCurve(paths.courbe, shock)
    short = None if paths.taux_court is None else paths.taux_court + shock
    return Paths(
        paths.dates,
        bumped_rates,
        factors * np.exp(-shock * elapsed[None, :]),
        bumped_curve,
        paths.parameters,
        short
    )

def _state_at_time(paths, values, t):
    elapsed = (pd.DatetimeIndex(paths.dates) - paths.dates[0]).days.to_numpy() / 365
    position = int(np.searchsorted(elapsed, t, side='left'))
    matrix = np.asarray(values)
    if position < len(elapsed) and abs(elapsed[position] - t) < 1e-12:
        return matrix[:, position]
    if position == 0:
        return matrix[:, 0]
    if position == len(elapsed):
        return matrix[:, -1]
    weight = (t - elapsed[position - 1]) / (elapsed[position] - elapsed[position - 1])
    return matrix[:, position - 1] + weight * (matrix[:, position] - matrix[:, position - 1])

def forward_rate(paths: Paths, fixing_date, payment_date):
    """Taux simple ACT/360 de la période exacte, y compris un stub final.

    Les chemins du générateur fournissent tous les états Hull–White nécessaires.
    Pour Paths manuels sans modèle, convention de taux zéro plat extrapolé du 6M.
    """
    date, end = (pd.Timestamp(fixing_date), pd.Timestamp(payment_date))
    dates = pd.DatetimeIndex(paths.dates)
    if date < dates[0] or date > dates[-1] or end <= date:
        raise ValueError('Dates de projection du coupon invalides.')
    elapsed = (dates - dates[0]).days.to_numpy() / 365
    t = (date - dates[0]).days / 365
    dt = (end - date).days / 365
    delta = (end - date).days / 360
    if paths.courbe is not None and paths.parameters is not None and (paths.taux_court is not None):
        a, sigma = (paths.parameters.a, paths.parameters.sigma)
        b = -np.expm1(-a * dt) / a
        r = _state_at_time(paths, paths.taux_court, t)
        c = paths.courbe
        bond = c.discount(t + dt) / c.discount(t) * np.exp(
            -b * (r - c.forward_instantane(t)) - sigma ** 2 / (4 * a) * -np.expm1(-2 * a * t) * b ** 2
        )
    else:
        rates = _state_at_time(paths, paths.taux_6m, t)
        six_days = (date + pd.DateOffset(months=6) - date).days
        bond = (1 + rates * six_days / 360) ** (-(end - date).days / six_days)
    return (1 / bond - 1) / delta

def forward_rate_for_period(paths, fixing_date, accrual_start, payment_date):
    """Fixing projeté à t du taux ACT/360 pour [S,T], avec t <= S < T."""
    fixing, start, end = map(pd.Timestamp, (fixing_date, accrual_start, payment_date))
    dates = pd.DatetimeIndex(paths.dates)
    if fixing < dates[0] or fixing > dates[-1] or start < fixing or (end <= start):
        raise ValueError('Dates de fixing ou de période invalides.')
    t = (fixing - dates[0]).days / 365
    s = (start - dates[0]).days / 365
    maturity = (end - dates[0]).days / 365
    delta = (end - start).days / 360
    elapsed = (dates - dates[0]).days.to_numpy() / 365
    if paths.courbe is not None and paths.parameters is not None and (paths.taux_court is not None):
        a, sigma, c = (paths.parameters.a, paths.parameters.sigma, paths.courbe)
        r = _state_at_time(paths, paths.taux_court, t)

        def price(T):
            b = -np.expm1(-a * (T - t)) / a
            return c.discount(T) / c.discount(t) * np.exp(
                -b * (r - c.forward_instantane(t)) - sigma ** 2 / (4 * a) * -np.expm1(-2 * a * t) * b * b
            )
        ratio = price(s) / price(maturity)
    else:
        six = _state_at_time(paths, paths.taux_6m, t)
        days = (fixing + pd.DateOffset(months=6) - fixing).days
        z = np.log1p(six * days / 360) / (days / 365)
        ratio = np.exp(z * (maturity - s))
    return (ratio - 1) / delta

def courbe_initiale(date, profil='plate', horizon=80.0):
    """Zéros continus : transition lisse vers le niveau long, échelle 10 ans.

    Les valeurs de PROFILS sont le taux à zéro et sa limite à long terme,
    pas des taux à 25 ans. Aucune observation ni estimation historique.
    """
    if profil not in PROFILS:
        raise ValueError(f'Profil inconnu : {profil}. Choisir parmi {tuple(PROFILS)}.')
    return SyntheticCurve(date, profil, horizon)

class SyntheticCurve(Curve):

    def __init__(self, date, profil, horizon):
        self.court, self.long = PROFILS[profil]
        super().__init__(date, np.array([0.0, horizon]), np.array([self.court, self.long]))

    def zero(self, t):
        t = np.asarray(t, dtype=float)
        if np.any(t < 0) or np.any(t > self.temps[-1]):
            raise ValueError('Maturité hors de la courbe synthétique.')
        return self.long + (self.court - self.long) * np.exp(-t / ECHELLE_COURBE)

    def forward_instantane(self, t):
        t = np.asarray(t, dtype=float)
        return self.zero(t) - t / ECHELLE_COURBE * (self.court - self.long) * np.exp(-t / ECHELLE_COURBE)

    def ancrer_sur_6m(self, date, taux_6m):
        return ShiftedCurve(self, date, taux_6m)

class ShiftedCurve(Curve):
    """Tarification simplifiée : conserver exactement la forme zéro initiale."""

    def __init__(self, base, date, six):
        self.base = base
        days = (pd.Timestamp(date) + pd.DateOffset(months=6) - pd.Timestamp(date)).days
        if 1 + six * days / 360 <= 0:
            raise ValueError('Taux 6M incompatible avec un facteur positif.')
        self.shift = np.log1p(six * days / 360) / (days / 365) - float(base.zero(days / 365))
        super().__init__(date, base.temps.copy(), base.taux_zero + self.shift)

    def zero(self, t):
        return self.base.zero(t) + self.shift

    def forward_instantane(self, t):
        return self.base.forward_instantane(t) + self.shift

class ParallelCurve(Curve):
    """Choc parallèle conservant la forme analytique de la courbe de base."""
    def __init__(self, base, shock):
        self.base = base
        self.shock = shock
        super().__init__(base.date, base.temps.copy(), base.taux_zero + shock)

    def zero(self, t):
        return self.base.zero(t) + self.shock

    def forward_instantane(self, t):
        return self.base.forward_instantane(t) + self.shock


class ConditionalCurve(Curve):
    """Courbe P(t,t+u) exacte du modèle initial, à l'état r(t) atteint."""

    def __init__(self, initial, date, short_rate, parameters):
        self.initial = initial
        self.elapsed = (pd.Timestamp(date) - initial.date).days / 365.0
        self.short_rate = float(short_rate)
        self.a = parameters.a
        self.c = parameters.sigma ** 2 / (4 * self.a) * -np.expm1(-2 * self.a * self.elapsed)
        self.shift = self.short_rate - float(initial.forward_instantane(self.elapsed))
        horizon = initial.temps[-1] - self.elapsed
        super().__init__(date, np.array([0.0, horizon]), np.array([short_rate, short_rate]))

    def discount(self, u):
        u = np.asarray(u, dtype=float)
        if np.any(u < 0) or np.any(u > self.temps[-1]):
            raise ValueError('Maturité hors de la courbe conditionnelle.')
        b = -np.expm1(-self.a * u) / self.a
        return self.initial.discount(self.elapsed + u) / self.initial.discount(self.elapsed) * np.exp(-b * self.shift - self.c * b * b)

    def zero(self, u):
        u = np.asarray(u, dtype=float)
        return -np.log(self.discount(u)) / np.where(u == 0, 1.0, u) + np.where(u == 0, self.short_rate, 0.0)

    def forward_instantane(self, u):
        u = np.asarray(u, dtype=float)
        b = -np.expm1(-self.a * u) / self.a
        return self.initial.forward_instantane(self.elapsed + u) + np.exp(-self.a * u) * (self.shift + 2 * self.c * b)

class SyntheticRepository:

    def __init__(self, settings, debut, profil):
        if settings.hw_mode != 'fixe':
            raise ValueError('Les profils synthétiques utilisent exclusivement hw_mode="fixe".')
        self.settings = settings
        self.initial = courbe_initiale(
            pd.Period(debut, freq='M').to_timestamp() + pd.offsets.MonthEnd(0),
            profil
        )
        end = self.initial.date + pd.DateOffset(months=6)
        six = float(
            (
                1 / self.initial.discount((end - self.initial.date).days / 365.0) - 1
            ) / ((end - self.initial.date).days / 360.0)
        )
        self.taux_historiques_6m = {self.initial.date: six}

    def courbe(self, mois):
        if mois != self.initial.date.strftime('%Y%m'):
            raise ValueError('Seule la courbe initiale est stockée ; les suivantes sont simulées.')
        return self.initial

    def historique(self, mois_fin):
        dates = pd.period_range(end=self.initial.date.to_period('M'), periods=7, freq='M')
        return pd.Series(self.taux_historiques_6m[self.initial.date], index=dates)

    def courbe_conditionnelle(self, date, central, parameters):
        i = central.dates.get_loc(date)
        return ConditionalCurve(self.initial, date, central.taux_court[0, i], parameters)
