"""Prêt fixe linéaire, prépaiement et quatre couvertures par swaps bullets."""
from dataclasses import dataclass, field, replace
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scripts.hw import (
    Settings, Paths, SyntheticRepository, generer_simulation,
    parallel_bump, forward_rate_for_period,
)

@dataclass(frozen=True)
class Loan:
    """Prêt fixe à amortissement linéaire."""
    id: str
    nominal_initial: float
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    amort_profile: pd.Series
    fixed_rate: float | None = None

    def __post_init__(self):
        start = pd.Timestamp(self.start_date).normalize()
        end = pd.Timestamp(self.end_date).normalize()
        object.__setattr__(self, 'start_date', start)
        object.__setattr__(self, 'end_date', end)
        if not self.id.strip():
            raise ValueError('Le prêt doit avoir un identifiant.')
        if not np.isfinite(self.nominal_initial) or self.nominal_initial <= 0:
            raise ValueError('Le nominal initial doit être strictement positif.')
        if end <= start:
            raise ValueError("La date d'échéance doit suivre la date de début.")
        profile = self.amort_profile
        if not isinstance(profile, pd.Series) or profile.empty:
            raise ValueError('amort_profile doit être une Series non vide.')
        if not isinstance(profile.index, pd.DatetimeIndex) or profile.index.has_duplicates:
            raise ValueError('Le profil doit avoir des dates uniques.')
        profile = profile.sort_index().astype(float)
        values = profile.to_numpy()
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError('Les encours doivent être finis et positifs ou nuls.')
        if np.any(values > self.nominal_initial * (1 + 1e-10)):
            raise ValueError('Un encours dépasse le nominal initial.')
        if np.any(np.diff(values) > self.nominal_initial * 1e-10):
            raise ValueError("Le profil d'encours doit être décroissant ou constant.")
        if profile.index[0].normalize() < start or profile.index[-1].normalize() > end:
            raise ValueError('Les dates du profil doivent rester dans la vie du prêt.')
        object.__setattr__(self, 'amort_profile', profile)
        if self.fixed_rate is None or not np.isfinite(self.fixed_rate):
            raise ValueError('Le taux fixe doit être fini.')

@dataclass(frozen=True)
class Swap:
    """Swap sans dépendance à QuantLib.

    notional est positif ; pay_fixed indique le sens de la position.
    """
    id: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    notional: float
    fixed_rate: float
    pay_fixed: bool
    trade_date: pd.Timestamp
    floating_fixings: dict | None = None
    closed_date: pd.Timestamp | None = None
    fixing_lag_business_days: int = 0
    floating_payment_months: int = 1

    def __post_init__(self):
        start = pd.Timestamp(self.start_date).normalize()
        end = pd.Timestamp(self.end_date).normalize()
        trade = pd.Timestamp(self.trade_date).normalize()
        object.__setattr__(self, 'start_date', start)
        object.__setattr__(self, 'end_date', end)
        object.__setattr__(self, 'trade_date', trade)
        if not self.id.strip() or end <= start:
            raise ValueError('Identifiant et dates de swap invalides.')
        if not np.isfinite(self.notional) or self.notional < 0:
            raise ValueError('Le nominal du swap doit être positif ou nul.')
        if not np.isfinite(self.fixed_rate):
            raise ValueError('Le taux fixe du swap doit être fini.')
        if not isinstance(self.fixing_lag_business_days, int) or self.fixing_lag_business_days < 0:
            raise ValueError('Le délai de fixing doit être un entier positif ou nul.')
        if self.floating_payment_months != 1:
            raise ValueError('Seuls les swaps à coupons mensuels sont utilisés ici.')
        fixings = {} if self.floating_fixings is None else {
            pd.Timestamp(d).normalize(): float(r) for d, r in self.floating_fixings.items()
        }
        if not all((np.isfinite(r) for r in fixings.values())):
            raise ValueError('Fixing non fini.')
        object.__setattr__(self, 'floating_fixings', fixings)
        if self.closed_date is not None:
            object.__setattr__(self, 'closed_date', pd.Timestamp(self.closed_date).normalize())

@dataclass(frozen=True)
class PrepaymentSettings:
    """Coefficients comportementaux, seuil de cumul et retrait almin."""
    intercept: float
    beta_change: float
    beta_gain: float
    minimum_rate: float
    liquidity_margin: float = 50 / 10000
    product_margin: float = 20 / 10000
    threshold: float = 0.5
    months_for_rate_change: int = 6
    refinancing_spread: float = 0.0
    refinancing_curve: object = field(default=None, repr=False, compare=False)
    almin: float = 0.004

DEFAULT_SETTINGS = PrepaymentSettings(
    intercept=-5,
    beta_change=4,
    beta_gain=3e-05,
    minimum_rate=0.003
)

@dataclass(frozen=True)
class PrepaymentPaths:
    """Probabilité, survie et date simulée de remboursement par chemin.

    Les matrices ont la forme (nombre de chemins, nombre de dates).
    prepay est persistant après le premier franchissement du seuil.
    """
    dates: pd.DatetimeIndex
    conditional_probability: np.ndarray
    cumulative_probability: np.ndarray
    survival_probability: np.ndarray
    prepay: np.ndarray
    offered_rates: np.ndarray
    reference_itp: np.ndarray
    spread_bps: np.ndarray

def add_prepayment(
    paths,
    loan: Loan,
    prior_6m_rates: pd.Series,
    settings: PrepaymentSettings=DEFAULT_SETTINGS
) -> PrepaymentPaths:
    """Probabilité conditionnelle, survie et cumul pour chaque chemin.

    Les observations antérieures servent au lag de variation des taux sur six
    mois. Elles ne servent à aucune calibration du modèle de taux.
    """
    if loan.fixed_rate is None:
        raise ValueError('Le modèle comportemental de prépaiement cible le prêt fixe.')
    parameters = [
        settings.intercept,
        settings.beta_change,
        settings.beta_gain,
        settings.minimum_rate,
        settings.liquidity_margin,
        settings.product_margin,
        settings.refinancing_spread
    ]
    if not np.all(np.isfinite(parameters)):
        raise ValueError('Paramètres de prépaiement non finis.')
    if not np.isfinite(settings.almin) or not 0 <= settings.almin <= 1:
        raise ValueError('almin doit être compris entre 0 et 1.')
    if settings.months_for_rate_change < 1:
        raise ValueError('Le décalage historique doit être positif.')
    if not 0 < settings.threshold < 1:
        raise ValueError('Le seuil doit être strictement compris entre 0 et 1.')
    if isinstance(paths, pd.DataFrame):
        if 'rate_6m' not in paths.columns:
            raise ValueError("Le scénario central doit contenir la colonne 'rate_6m'.")
        dates = pd.DatetimeIndex(paths.index)
        rates = paths['rate_6m'].to_numpy(dtype=float)[None, :]
    else:
        dates = pd.DatetimeIndex(paths.dates)
        rates = np.asarray(paths.six_month_rates, dtype=float)
    if len(dates) == 0 or not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError('Dates des chemins invalides.')
    if rates.ndim != 2 or rates.shape[1] != len(dates) or rates.shape[0] == 0:
        raise ValueError('Les chemins doivent avoir la forme (chemins, dates).')
    if not np.all(np.isfinite(rates)):
        raise ValueError('Les chemins contiennent un taux non fini.')
    if not isinstance(prior_6m_rates, pd.Series) or prior_6m_rates.empty:
        raise ValueError("Fournir l'historique mensuel des taux 6M avant la projection.")
    prior = prior_6m_rates.astype(float).copy()
    prior.index = pd.DatetimeIndex(prior.index).to_period('M')
    prior = prior.groupby(level=0).last()
    months = dates.to_period('M')
    forecast_index = {month: i for i, month in enumerate(months)}
    lag_rates = np.empty_like(rates)
    lag_months = settings.months_for_rate_change
    for j, month in enumerate(months):
        lag_month = month - lag_months
        if lag_month in forecast_index and forecast_index[lag_month] < j:
            lag_rates[:, j] = rates[:, forecast_index[lag_month]]
        elif lag_month in prior.index:
            lag_rates[:, j] = float(prior.loc[lag_month])
        else:
            raise ValueError(f'Taux 6M historique manquant pour {lag_month}.')
    dates_restantes = np.maximum((loan.end_date - dates).days.to_numpy(dtype=float) / 365, 0)
    refi_now = rates + settings.liquidity_margin
    refi_then = lag_rates + settings.liquidity_margin
    if settings.refinancing_curve is None:
        raise ValueError('Fournir la courbe de refinancement : le taux 6M seul ne remplace pas l’ITP.')
    monthly = dates.isin(loan.amort_profile.index) & (dates < loan.end_date)
    source = getattr(paths, 'refinancing_source_paths', None)
    key = (
        id(settings.refinancing_curve),
        loan.end_date,
        settings.liquidity_margin,
        settings.product_margin,
        settings.refinancing_spread,
        settings.minimum_rate
    )
    cache = getattr(source, '_refinancing_tariffs', {}) if source is not None else {}
    if key not in cache:
        nouveau_tarif = np.full_like(rates, np.inf)
        reference_itp = np.full_like(rates, np.nan)
        nouveau_tarif[:, monthly], reference_itp[:, monthly] = refinancing_tariffs(
            dates[monthly],
            rates[:, monthly],
            loan.end_date,
            settings.refinancing_curve,
            settings.liquidity_margin + settings.product_margin + settings.refinancing_spread,
            settings.minimum_rate,
            return_itp=True
        )
        if source is not None:
            cache[key] = (nouveau_tarif, reference_itp)
            source._refinancing_tariffs = cache
    else:
        nouveau_tarif, reference_itp = cache[key]
    spread_bps = (loan.fixed_rate - nouveau_tarif) * 10000
    signed_spread = np.where(np.isfinite(spread_bps), spread_bps, 0.0)
    variation_taux = signed_spread * np.abs(refi_now - refi_then)
    gain_refinancement = signed_spread * dates_restantes[None, :]
    score = settings.intercept + settings.beta_change * variation_taux + settings.beta_gain * gain_refinancement
    score = np.clip(score, -700, 700)
    logistique = 1 / (1 + np.exp(-score))
    with np.errstate(divide='ignore'):
        probability = -np.expm1(3 * np.log1p(-logistique))
    # almin est soustrait après la transformation en probabilité.
    probability = np.maximum(probability - settings.almin, 0.0)
    eligible = dates.isin(loan.amort_profile.index) & (dates > dates[0]) & (dates < loan.end_date)
    probability[:, ~eligible] = 0
    probability = np.clip(probability, 0, 1)
    with np.errstate(divide='ignore'):
        log_survival = np.cumsum(np.log1p(-probability), axis=1)
    survival = np.exp(log_survival)
    cumulative = -np.expm1(log_survival)
    prepay = np.maximum.accumulate(cumulative >= settings.threshold, axis=1)
    return PrepaymentPaths(
        dates,
        probability,
        cumulative,
        survival,
        prepay,
        np.where(np.isfinite(nouveau_tarif), nouveau_tarif, np.nan),
        reference_itp,
        np.where(np.isfinite(spread_bps), spread_bps, np.nan)
    )

def payment_dates(start_date, end_date):
    start, end = (pd.Timestamp(start_date).normalize(), pd.Timestamp(end_date).normalize())
    if end <= start:
        raise ValueError('La maturité doit suivre le départ.')
    dates = list(pd.date_range(start + pd.offsets.MonthEnd(1), end, freq='ME'))
    if not dates or dates[-1] != end:
        dates.append(end)
    return pd.DatetimeIndex(dates)

def build_amortization_profile(start_date, end_date, nominal, fixed_rate=0.0):
    """Encours avant échéance : capital constant N/n, indépendant du coupon."""
    dates = payment_dates(start_date, end_date)
    if nominal <= 0 or not np.isfinite(nominal):
        raise ValueError('Nominal invalide.')
    opening = float(nominal) * (len(dates) - np.arange(len(dates))) / len(dates)
    return pd.Series(opening, index=dates, name='outstanding')

def fixed_cashflows(loan: Loan) -> pd.DataFrame:
    """Capital mensuel constant et intérêts r/12 sur l'encours avant paiement."""
    if loan.fixed_rate is None:
        raise ValueError('Ce dossier est réservé aux prêts à taux fixe.')
    balance = loan.amort_profile.sort_index()
    opening = balance.to_numpy(dtype=float)
    closing = np.r_[opening[1:], 0.0]
    principal = opening - closing
    interest = opening * loan.fixed_rate / 12
    total = principal + interest
    if not np.allclose(principal, loan.nominal_initial / len(opening), rtol=1e-09, atol=1e-07):
        raise ValueError('Le profil ne correspond pas à un amortissement linéaire.')
    return pd.DataFrame(
        {
            'outstanding': opening,
            'closing_outstanding': closing,
            'principal': principal,
            'interest': interest,
            'cashflow': total
        },
        index=balance.index
    )

def outstanding_at(loan: Loan, date, *, after_payment=True):
    """Encours vivant avant/après paiement ; PV calculées après paiement."""
    date = pd.Timestamp(date).normalize()
    cf = fixed_cashflows(loan)
    if after_payment:
        paid = cf.loc[cf.index <= date]
    else:
        paid = cf.loc[cf.index < date]
    return float(paid['closing_outstanding'].iloc[-1]) if not paid.empty else float(loan.nominal_initial)

def _prepayment_for_hw_paths(
    paths: Paths,
    loan: Loan,
    prior_6m_rates: pd.Series,
    settings: PrepaymentSettings
):
    """Adapte Paths.taux_6m au nom d'attribut attendu par prepayment.py."""
    rates_view = SimpleNamespace(
        dates=paths.dates,
        six_month_rates=paths.taux_6m,
        refinancing_source_paths=paths
    )
    return add_prepayment(rates_view, loan, prior_6m_rates, settings)

def _value_on_paths(
    loan: Loan,
    paths: Paths,
    prior_6m_rates: pd.Series,
    prepayment_fee: float,
    prepayment_settings: PrepaymentSettings,
    prepayment_override=None
) -> np.ndarray:
    if prepayment_fee < 0 or not np.isfinite(prepayment_fee):
        raise ValueError("L'indemnité de prépaiement doit être finie et positive ou nulle.")
    dates = pd.DatetimeIndex(paths.dates).normalize()
    if not loan.start_date <= dates[0] < loan.end_date:
        raise ValueError('La date de valorisation doit être dans la vie du prêt.')
    prepayment = prepayment_override if prepayment_override is not None else _prepayment_for_hw_paths(paths, loan, prior_6m_rates, prepayment_settings)
    contract = fixed_cashflows(loan)
    rates = np.asarray(paths.taux_6m, dtype=float)
    discount_factors = np.asarray(paths.facteurs_actualisation, dtype=float)
    if rates.ndim != 2 or rates.shape[1] != len(dates):
        raise ValueError('Paths.taux_6m doit avoir la forme (chemins, dates).')
    expected_shape = (rates.shape[0], len(dates))
    if discount_factors.shape != expected_shape:
        raise ValueError('Paths.facteurs_actualisation doit avoir la même forme que taux_6m.')
    columns = {date: i for i, date in enumerate(dates)}
    values = np.zeros(rates.shape[0])
    for date, cashflow in contract.iterrows():
        payment_date = pd.Timestamp(date).normalize()
        if payment_date <= dates[0]:
            continue
        if payment_date not in columns:
            raise ValueError(
                f"Le paiement du {payment_date.date()} n'est pas dans les dates Hull–White."
            )
        k = columns[payment_date]
        conditional_probability = prepayment.conditional_probability[:, k]
        if k == 0:
            survival_before = np.ones(rates.shape[0])
        else:
            survival_before = prepayment.survival_probability[:, k - 1]
        probability_of_prepayment = survival_before * conditional_probability
        expected_cashflow = survival_before * cashflow['cashflow'] + probability_of_prepayment * cashflow['closing_outstanding'] * (1 + prepayment_fee)
        values += expected_cashflow * discount_factors[:, k]
    return values

def pv_fixed_loan(
    loan: Loan,
    paths: Paths,
    prior_6m_rates: pd.Series,
    *,
    prepayment_fee: float=0.0,
    prepayment_settings: PrepaymentSettings=DEFAULT_SETTINGS
) -> np.ndarray:
    """Renvoie la valeur actuelle du prêt pour chacun des chemins Hull–White."""
    return _value_on_paths(loan, paths, prior_6m_rates, prepayment_fee, prepayment_settings)

def _parallel_bump(paths: Paths, shock: float) -> Paths:
    """Choc de taux zéro continu ; projection 6M convertie exactement ACT/360."""
    return parallel_bump(paths, shock)

def bpv_fixed_loan(
    loan: Loan,
    paths: Paths,
    prior_6m_rates: pd.Series,
    *,
    shock: float=0.0001,
    prepayment_fee: float=0.0,
    prepayment_settings: PrepaymentSettings=DEFAULT_SETTINGS
) -> dict:
    """Calcule la PV et la BPV après un choc parallèle de la courbe zéro.

    Convention : BPV = PV après une hausse de 1 bp - PV de base.
    Le calcul réutilise les mêmes chemins et recalcule le prépaiement.
    """
    if not np.isfinite(shock) or shock <= 0:
        raise ValueError('Le choc doit être positif et fini.')
    bumped_paths = _parallel_bump(paths, shock)
    pv_base_paths = pv_fixed_loan(
        loan,
        paths,
        prior_6m_rates,
        prepayment_fee=prepayment_fee,
        prepayment_settings=prepayment_settings
    )
    pv_bumped_paths = pv_fixed_loan(
        loan,
        bumped_paths,
        prior_6m_rates,
        prepayment_fee=prepayment_fee,
        prepayment_settings=prepayment_settings
    )
    prepayment_base = _prepayment_for_hw_paths(paths, loan, prior_6m_rates, prepayment_settings)
    pv_frozen_paths = _value_on_paths(
        loan,
        bumped_paths,
        prior_6m_rates,
        prepayment_fee,
        prepayment_settings,
        prepayment_override=prepayment_base
    )
    scale = 0.0001 / shock
    bpv_total = float((pv_bumped_paths - pv_base_paths).mean()) * scale
    bpv_frozen = float((pv_frozen_paths - pv_base_paths).mean()) * scale
    sample_count = len(pv_base_paths)

    def standard_error(values):
        return float(np.std(values, ddof=1) / np.sqrt(sample_count)) if sample_count > 1 else 0.0
    return {
        'number_of_paths': sample_count,
        'pv_std_error': standard_error(pv_base_paths),
        'bpv_std_error': standard_error(pv_bumped_paths - pv_base_paths) * scale,
        'bpv_frozen': bpv_frozen,
        'bpv_prepayment_effect': bpv_total - bpv_frozen,
        'pv': float(pv_base_paths.mean()),
        'pv_shocked': float(pv_bumped_paths.mean()),
        'bpv': bpv_total,
        'pv_by_path': pv_base_paths,
        'pv_shocked_by_path': pv_bumped_paths
    }

def expected_remaining_life(loan, paths, prior_6m_rates, settings=DEFAULT_SETTINGS):
    """E[min(date de prépaiement, maturité) - date actuelle], non pondérée par capital.

    Calculée uniquement sur les futurs conditionnels connus à la date actuelle.
    Ce n'est ni une durée de Macaulay ni une WAL des remboursements de principal.
    """
    pp = _prepayment_for_hw_paths(paths, loan, prior_6m_rates, settings)
    marginal = np.c_[
        np.zeros(len(pp.survival_probability)),
        pp.survival_probability[:, :-1] - pp.survival_probability[:, 1:]
    ]
    years = (pd.DatetimeIndex(paths.dates) - paths.dates[0]).days.to_numpy() / 365
    life = (marginal * years[None, :]).sum(axis=1) + pp.survival_probability[:, -1] * years[-1]
    return {'expected_years': float(life.mean()), 'expected_years_by_path': life}

def _month_end_schedule(start_date, end_date, step_months):
    start, end = (pd.Timestamp(start_date).normalize(), pd.Timestamp(end_date).normalize())
    if end <= start:
        raise ValueError('Échéance de swap invalide.')
    dates = list(
        pd.date_range(
            start + pd.offsets.MonthEnd(step_months),
            end,
            freq=pd.offsets.MonthEnd(step_months)
        )
    )
    if not dates or dates[-1] != end:
        dates.append(end)
    return pd.DatetimeIndex(dates)

def _periods(swap, step):
    starts, ends = _schedule_arrays(swap.start_date, swap.end_date, step)
    yield from zip(starts, ends)

def _path_values_at(paths, dates, values, logarithmic=False):
    known, requested = (pd.DatetimeIndex(paths.dates), pd.DatetimeIndex(dates))
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(known):
        raise ValueError('Matrice de chemins invalide.')
    if np.any(requested < known[0]) or np.any(requested > known[-1]):
        raise ValueError("Interpolation hors de l'horizon des chemins.")
    exact = known.get_indexer(requested)
    if np.all(exact >= 0):
        result = matrix[:, exact]
        if logarithmic and np.any(result <= 0):
            raise ValueError("Facteur d'actualisation non positif.")
        return result
    x = (known - known[0]).days.to_numpy(dtype=float)
    targets = (requested - known[0]).days.to_numpy(dtype=float)
    right = np.clip(np.searchsorted(x, targets), 1, len(x) - 1)
    left = right - 1
    weight = (targets - x[left]) / (x[right] - x[left])
    a, b = (matrix[:, left], matrix[:, right])
    if logarithmic:
        if np.any(a <= 0) or np.any(b <= 0):
            raise ValueError("Facteur d'actualisation non positif.")
        return np.exp(np.log(a) + weight[None, :] * (np.log(b) - np.log(a)))
    return a + weight[None, :] * (b - a)

def notional_at(swap, date):
    date = pd.Timestamp(date)
    if date < swap.start_date or date >= swap.end_date:
        return 0.0
    return float(swap.notional)

def _notional_accrual(swap, start, end, denominator):
    """Intégrale du notionnel par morceaux ; variations après paiement du prêt."""
    start, end = (max(pd.Timestamp(start), swap.start_date), min(pd.Timestamp(end), swap.end_date))
    if end <= start:
        return 0.0
    return float(swap.notional) * (end - start).days / denominator

@lru_cache(maxsize=8192)
def _schedule_arrays(start, end, step):
    payments = _month_end_schedule(start, end, step)
    starts = pd.DatetimeIndex([start]).append(payments[:-1])
    return (starts, payments)

def _forward_rates_vector(paths, fixing, start, end):
    t = (fixing - paths.dates[0]).days.to_numpy() / 365
    dt_start = (start - fixing).days.to_numpy() / 365
    dt_end = (end - fixing).days.to_numpy() / 365
    delta = (end - start).days.to_numpy() / 360
    if np.any(fixing < paths.dates[0]) or np.any(start < fixing):
        raise ValueError('Fixing historique manquant ou date de projection invalide.')
    if paths.courbe is not None and paths.parameters is not None and (paths.taux_court is not None):
        c = paths.courbe
        a, sigma = (paths.parameters.a, paths.parameters.sigma)
        r = _path_values_at(paths, fixing, paths.taux_court)
        b_start = -np.expm1(-a * dt_start) / a
        b_end = -np.expm1(-a * dt_end) / a
        ratio = (c.discount(t + dt_start) / c.discount(t + dt_end))[None, :] * np.exp(
            (b_end - b_start)[None, :] * (r - c.forward_instantane(t)[None, :]) + sigma ** 2 / (4 * a) * (-np.expm1(-2 * a * t))[None, :] * (b_end * b_end - b_start * b_start)[None, :]
        )
    else:
        six = _path_values_at(paths, fixing, paths.taux_6m)
        days = np.array([(d + pd.DateOffset(months=6) - d).days for d in fixing])
        zero = np.log1p(six * days[None, :] / 360) / (days[None, :] / 365)
        ratio = np.exp(zero * (dt_end - dt_start)[None, :])
    return (ratio - 1) / delta[None, :]

def pv_swap(swap, paths):
    """Valorisation d'un bullet mensuel, après les coupons du jour."""
    valuation = paths.dates[0]
    if (swap.trade_date > valuation or valuation >= swap.end_date
            or (swap.closed_date is not None and swap.closed_date <= valuation)):
        return np.zeros(len(paths.taux_6m))
    start = max(swap.start_date, valuation)
    known_fixings = tuple(sorted((d, r) for d, r in swap.floating_fixings.items() if d >= start))
    key = (start, swap.fixing_lag_business_days, known_fixings)
    cache = getattr(paths, '_swap_legs', None)
    if cache is None:
        cache = {}
        paths._swap_legs = cache
    if key not in cache:
        starts, ends = _schedule_arrays(start, paths.dates[-1], 1)
        factors = _path_values_at(paths, ends, paths.facteurs_actualisation, True)
        rates = np.empty((len(paths.taux_6m), len(ends)))
        known = np.array([d in swap.floating_fixings for d in starts])
        if known.any():
            rates[:, known] = [swap.floating_fixings[d] for d in starts[known]]
        if (~known).any():
            projected = starts[~known]
            fixing = projected if swap.fixing_lag_business_days == 0 else projected - pd.offsets.BDay(swap.fixing_lag_business_days)
            rates[:, ~known] = _forward_rates_vector(paths, fixing, projected, ends[~known])
        days = (ends - starts).days.to_numpy()
        fixed = np.cumsum(factors * (days / 365)[None, :], axis=1)
        floating = np.cumsum(factors * rates * (days / 360)[None, :], axis=1)
        cache[key] = (ends, fixed, floating)
    ends, fixed, floating = cache[key]
    column = ends.get_indexer([swap.end_date])[0]
    if column < 0:
        raise ValueError('La fin du swap doit être une échéance mensuelle du prêt.')
    sign = 1 if swap.pay_fixed else -1
    return sign * swap.notional * (floating[:, column] - swap.fixed_rate * fixed[:, column])


def _cached_bump(paths, shock):
    cache = getattr(paths, '_swap_bumps', None)
    if cache is None:
        cache = {}
        paths._swap_bumps = cache
    if shock not in cache:
        cache[shock] = parallel_bump(paths, shock)
    return cache[shock]

def clear_pricing_cache(paths):
    """Libère les caches dès que le mois est terminé, pour limiter la mémoire."""
    if paths is not None:
        for attribute in ['_swap_bumps', '_swap_legs']:
            if hasattr(paths, attribute):
                delattr(paths, attribute)

def bpv_swap(swap, paths, shock=0.0001):
    if not np.isfinite(shock) or shock <= 0:
        raise ValueError('Le choc doit être positif.')
    base, up = (pv_swap(swap, paths), pv_swap(swap, _cached_bump(paths, shock)))
    return {
        'pv': float(base.mean()),
        'pv_shocked': float(up.mean()),
        'bpv': float((up - base).mean()) * 0.0001 / shock,
        'pv_by_path': base,
        'pv_shocked_by_path': up
    }

def _par_rate_for_swap(swap, paths):
    floating = replace(swap, fixed_rate=0.0)
    zero = float(pv_swap(floating, paths).mean())
    one = float(pv_swap(replace(swap, fixed_rate=1.0), paths).mean())
    annuity = zero - one if swap.pay_fixed else one - zero
    if annuity <= 0:
        raise ValueError('Annuité du swap nulle.')
    return zero / annuity if swap.pay_fixed else -zero / annuity

def build_bpv_hedge_swap(swap_id, loan_bpv, paths, end_date, *, shock=0.0001, start_lag=4, fixing_lag=2):
    """Prendre un bullet au pair dont la BPV compense celle du portefeuille."""
    date = pd.Timestamp(paths.dates[0])
    start = date if start_lag == 0 else date + pd.offsets.BDay(start_lag)
    if not np.isfinite(loan_bpv):
        raise ValueError('BPV non finie.')
    if start >= pd.Timestamp(end_date):
        raise ValueError("La maturité précède la prise d'effet du swap.")
    unit = Swap(
        swap_id,
        start,
        end_date,
        1.0,
        0.0,
        True,
        date,
        floating_fixings={},
        fixing_lag_business_days=fixing_lag
    )
    if start_lag == 0 and fixing_lag == 0:
        first_payment = payment_dates(start, end_date)[0]
        fixing = float(forward_rate_for_period(paths, date, start, first_payment)[0])
        unit = replace(unit, floating_fixings={start: fixing})
    unit = replace(unit, fixed_rate=_par_rate_for_swap(unit, paths))
    unit_bpv = bpv_swap(unit, paths, shock)['bpv']
    if abs(unit_bpv) < 1e-12:
        raise ValueError('BPV unitaire trop faible pour couvrir le prêt.')
    signed_notional = -loan_bpv / unit_bpv
    swap = replace(unit, notional=abs(signed_notional), pay_fixed=signed_notional >= 0)
    residual = loan_bpv + bpv_swap(swap, paths, shock)['bpv']
    if abs(residual) > 1e-06:
        raise ArithmeticError('La couverture ne neutralise pas la BPV.')
    return swap

@lru_cache(maxsize=8192)
def fixing_schedule(start, end, lag):
    starts, ends = _schedule_arrays(start, end, 1)
    fixing_dates = starts if lag == 0 else starts - pd.offsets.BDay(lag)
    return starts, ends, fixing_dates


def update_known_fixings(swaps, date, simulation):
    """Figer seulement les nouveaux coupons dont le fixing est déjà connu."""
    date = pd.Timestamp(date)
    for key, swap in list(swaps.items()):
        if swap.trade_date > date or (swap.closed_date is not None and swap.closed_date <= date):
            continue
        starts, ends, fixing_dates = fixing_schedule(swap.start_date, swap.end_date,
                                                     swap.fixing_lag_business_days)
        last = fixing_dates.searchsorted(date, side='right')
        missing = [i for i in range(last) if starts[i] not in swap.floating_fixings]
        if not missing:
            continue
        fixings = dict(swap.floating_fixings)
        for i in missing:
            fixings[starts[i]] = float(forward_rate_for_period(
                simulation.central_paths, fixing_dates[i], starts[i], ends[i])[0])
        swaps[key] = replace(swap, floating_fixings=fixings)


def _known_central_rate(swap, start, end, scenario):
    if start in swap.floating_fixings:
        return swap.floating_fixings[start]
    fixing = start if swap.fixing_lag_business_days == 0 else start - pd.offsets.BDay(swap.fixing_lag_business_days)
    return _rate_at_date(scenario, fixing)

def swap_cash_income(swap, period_start, period_end, central_scenario):
    begin, finish = (pd.Timestamp(period_start), pd.Timestamp(period_end))
    if swap.trade_date > finish:
        return 0.0
    sign = 1 if swap.pay_fixed else -1
    income = 0.0
    for fixed_leg, step, denominator in [(True, 1, 365), (False, swap.floating_payment_months, 360)]:
        starts, ends = _schedule_arrays(swap.start_date, swap.end_date, step)
        left = ends.searchsorted(begin, side='right')
        right = ends.searchsorted(finish, side='right')
        for start, end in zip(starts[left:right], ends[left:right]):
            if begin < end <= finish and (swap.closed_date is None or end <= swap.closed_date):
                rate = -swap.fixed_rate if fixed_leg else _known_central_rate(swap, start, end, central_scenario)
                income += sign * rate * _notional_accrual(swap, start, end, denominator)
    return float(income)

def committed_notional_at(swap, date):
    """Inclut le notionnel déjà engagé d'un swap dont la prise d'effet est future."""
    date = pd.Timestamp(date)
    if (
        swap.trade_date > date
        or date >= swap.end_date
        or swap.closed_date is not None and swap.closed_date <= date
    ):
        return 0.0
    return float(swap.notional) if date < swap.start_date else notional_at(swap, date)

def calculate_itp(curve, amort_profile: pd.Series) -> dict:
    """Résout PV(capital) + ITP * PV(encours/12) = nominal."""
    if amort_profile.empty:
        raise ValueError('Échéancier vide.')
    dates = pd.DatetimeIndex(amort_profile.sort_index().index)
    nominal = float(amort_profile.sort_index().iloc[0])
    times = (dates - pd.Timestamp(curve.date)).days.to_numpy() / 365
    if np.any(times <= 0):
        raise ValueError('Toutes les échéances doivent suivre la date de courbe.')
    df = np.asarray(curve.discount(times), dtype=float)
    if df.shape != times.shape or not np.all(np.isfinite(df)) or np.any(df <= 0):
        raise ValueError("Facteurs d'actualisation invalides.")
    opening = amort_profile.sort_index().to_numpy(dtype=float)
    principal = opening - np.r_[opening[1:], 0.0]
    if np.any(principal <= 0) or not np.all(np.isfinite(opening)):
        raise ValueError('Profil amortissant invalide.')
    pv_principal = float(np.dot(principal, df))
    annuity = float(np.dot(opening / 12, df))
    rate = (nominal - pv_principal) / annuity
    return {
        'annual_nominal_itp': float(rate),
        'annual_effective_itp': float(np.expm1(12 * np.log1p(rate / 12))),
        'quarterly_itp': float(np.expm1(3 * np.log1p(rate / 12))),
        'initial_monthly_payment_at_itp': float(principal[0] + opening[0] * rate / 12),
        'pv_principal': pv_principal,
        'interest_annuity': annuity,
        'pv_at_itp': pv_principal + rate * annuity
    }

def build_loan_with_itp(
    loan_id,
    nominal_initial,
    start_date,
    end_date,
    curve,
    *,
    commercial_margin=20 / 10000,
    liquidity_margin=50 / 10000,
    refinancing_spread=0.0,
    minimum_rate=None
):
    if not np.all(np.isfinite([commercial_margin, liquidity_margin, refinancing_spread])):
        raise ValueError('Marges non finies.')
    profile = build_amortization_profile(start_date, end_date, nominal_initial)
    itp = calculate_itp(curve, profile)
    fixed_rate = itp['annual_nominal_itp'] + liquidity_margin + commercial_margin + refinancing_spread
    if minimum_rate is not None:
        fixed_rate = max(fixed_rate, minimum_rate)
    return Loan(loan_id, nominal_initial, start_date, end_date, profile, fixed_rate=fixed_rate)

def refinancing_tariffs(dates, rates, end, curve, margin, minimum_rate, *, return_itp=False):
    """Même ITP et tarif que build_loan_with_itp, vectorisés sur dates et chemins."""
    dates = pd.DatetimeIndex(dates)
    rates = np.asarray(rates, dtype=float)
    eligible = np.flatnonzero(dates < pd.Timestamp(end))
    result = np.full_like(rates, np.inf)
    itps = np.full_like(rates, np.nan)
    for col, j in enumerate(eligible):
        date = dates[j]
        payments = payment_dates(date, end)
        times = (payments - date).days.to_numpy() / 365
        six_days = (date + pd.DateOffset(months=6) - date).days
        if np.any(1 + rates[:, j] * six_days / 360 <= 0):
            raise ValueError('Taux 6M incompatible avec un DF positif.')
        zero = np.log1p(rates[:, j] * six_days / 360) / (six_days / 365)
        shift = zero - float(curve.zero(six_days / 365))
        factors = np.exp(-(curve.zero(times)[None, :] + shift[:, None]) * times[None, :])
        count = len(payments)
        opening = (count - np.arange(count)) / count
        itps[:, j] = (1 - factors.sum(axis=1) / count) / (factors @ opening / 12)
        result[:, j] = np.maximum(itps[:, j] + margin, minimum_rate)
    return (result, itps) if return_itp else result

def _rate_at_date(scenario, date):
    column = next((c for c in ('taux_6m', '6', 'rate_6m') if c in scenario), None)
    if column is None:
        raise ValueError('Colonne de taux 6M absente.')
    known = scenario[column].sort_index().loc[:pd.Timestamp(date)]
    if known.empty:
        raise ValueError(f'Aucun taux connu au {date}.')
    return float(known.iloc[-1])

def compute_carry_for_period(loan, period_start, period_end, central_scenario, swaps, *,
                             liquidity_margin=.005, opening_outstanding=None,
                             loan_interest=None, swap_cashflow_eur=None):
    """Encaissements du mois, hors principal et hors valorisation non réalisée."""
    opening = outstanding_at(loan, period_start) if opening_outstanding is None else float(opening_outstanding)
    interest = opening * loan.fixed_rate / 12 if loan_interest is None else float(loan_interest)
    funding = opening * _rate_at_date(central_scenario, period_start) / 12
    liquidity = opening * liquidity_margin / 12
    if swap_cashflow_eur is None:
        swap_cashflow_eur = sum(swap_cash_income(swap, period_start, period_end, central_scenario)
                               for swap in swaps.values())
    return {
        'opening_outstanding': opening,
        'loan_interest_eur': interest,
        'funding_cost_eur': funding,
        'liquidity_cost_eur': liquidity,
        'swap_cashflow_eur': swap_cashflow_eur,
    }


STRATEGIES = [
    ('fin_contractuelle_unwind', False, True),
    ('fin_contractuelle_conservation', False, False),
    ('fin_esperee_unwind', True, True),
    ('fin_esperee_conservation', True, False)
]

def market(parameters):
    start = pd.Period(parameters['debut'], freq='M').to_timestamp(how='end').normalize()
    end = start + pd.DateOffset(years=parameters['maturite_ans'])
    settings = Settings(
        nombre_chemins=parameters['chemins_conditionnels'],
        graine=parameters['seed'],
        hw_mode='fixe',
        hw_a=parameters.get('hw_a', 0.1),
        hw_sigma=parameters.get('hw_sigma', 0.005)
    )
    repo = SyntheticRepository(settings, parameters['debut'], parameters.get('profil_initial', 'plate'))
    return (repo, generer_simulation(repo, parameters['debut'], end))

def prior(history, central, date):
    history = history.copy()
    history.index = history.index.to_timestamp(how='end').normalize() if isinstance(history.index, pd.PeriodIndex) else pd.DatetimeIndex(history.index)
    both = pd.concat([history[history.index < date], central.loc[central.index < date, 'taux_6m']])
    monthly = both.groupby(both.index.to_period('M')).last()
    monthly.index = monthly.index.to_timestamp(how='end').normalize()
    return monthly

def prepayment_date(loan, simulation, history, settings):
    rates = simulation.scenario_central.loc[loan.start_date:, ['taux_6m']].rename(columns={'taux_6m': 'rate_6m'})
    table = add_prepayment(
        rates,
        loan,
        prior(history, simulation.scenario_central, loan.start_date),
        settings
    )
    crossed = table.prepay[0] & (table.dates > loan.start_date) & (table.dates < loan.end_date)
    return (table.dates[crossed][0] if crossed.any() else None, table)

def strategy(
    parameters,
    context,
    name,
    expected,
    unwind,
    *,
    prepayment_settings=DEFAULT_SETTINGS,
    save_swaps=True
):
    repo, simulation = context
    central = simulation.scenario_central
    start, end = (central.index[0], central.index[-1])
    curve = repo.courbe(parameters['debut'])
    history = repo.historique(parameters['debut'])
    settings = replace(
        prepayment_settings,
        product_margin=parameters['marge_commerciale'],
        liquidity_margin=parameters['marge_liquidite'],
        refinancing_spread=parameters['marge_refinancement'],
        refinancing_curve=curve
    )

    def create_loan(identifier, nominal, date, refinancing=0.0):
        current = curve.ancrer_sur_6m(date, float(central.loc[date, 'taux_6m']))
        return build_loan_with_itp(
            identifier,
            nominal,
            date,
            end,
            current,
            commercial_margin=parameters['marge_commerciale'],
            liquidity_margin=parameters['marge_liquidite'],
            refinancing_spread=refinancing,
            minimum_rate=settings.minimum_rate
        )
    loan = create_loan('pret_1', parameters['nominal'], start)
    event, prepayment_trace = prepayment_date(loan, simulation, history, settings)
    loan_count = 1
    swaps = {}
    monthly = []
    audit = []
    previous = None
    cumulative = 0.0
    for date in central.index:
        paths = simulation.scenarios_conditionnels.get(date.strftime('%Y%m'))
        update_known_fixings(swaps, date, simulation)
        old_loan = loan
        old_trace = prepayment_trace
        principal = settlement = fee = 0.0
        carry = None
        coupon_cash = {
            key: swap_cash_income(s, previous, date, central) if previous is not None else 0.0 for key, s in swaps.items()
        }
        if previous is not None:
            payment = fixed_cashflows(loan).loc[date]
            carry = compute_carry_for_period(
                loan,
                previous,
                date,
                central,
                swaps,
                liquidity_margin=parameters['marge_liquidite'],
                opening_outstanding=float(payment.outstanding),
                loan_interest=float(payment.interest),
                swap_cashflow_eur=sum(coupon_cash.values())
            )
        values = {
            key: bpv_swap(s, paths) if paths is not None else {'pv': 0.0, 'bpv': 0.0} for key, s in swaps.items()
        }
        closed = {}
        # Paiement normal d'abord, puis prépaiement et remplacement du prêt.
        if date == event:
            principal = outstanding_at(loan, date)
            fee = principal * parameters['indemnite_prepaiement']
            if unwind:
                for key, s in list(swaps.items()):
                    if s.closed_date is None and s.trade_date <= date < s.end_date:
                        closed[key] = values[key]['pv']
                        settlement += closed[key]
                        swaps[key] = replace(s, closed_date=date)
                        values[key] = {'pv': 0.0, 'bpv': 0.0}
            loan_count += 1
            loan = create_loan(f'pret_{loan_count}', principal, date, parameters['marge_refinancement'])
            event, prepayment_trace = prepayment_date(loan, simulation, history, settings)
        risk = bpv_fixed_loan(
            loan,
            paths,
            prior(history, central, date),
            prepayment_settings=settings,
            prepayment_fee=parameters['indemnite_prepaiement']
        ) if paths is not None else {'bpv': 0.0, 'pv': 0.0}
        # Couvrir le prêt et tous les swaps encore détenus, avec un nouveau bullet.
        residual = risk['bpv'] + sum((v['bpv'] for v in values.values()))
        swap_end = end
        if expected and paths is not None:
            life = expected_remaining_life(loan, paths, prior(history, central, date), settings)['expected_years']
            candidates = loan.amort_profile.index[loan.amort_profile.index > date]
            swap_end = candidates[np.argmin(np.abs((candidates - date).days.to_numpy() / 365 - life))]
        if abs(residual) > 1e-08:
            if paths is None or date + pd.offsets.BDay(4) >= swap_end:
                raise ArithmeticError(f'Pas de swap disponible pour neutraliser la BPV au {date}.')
            new = build_bpv_hedge_swap(f'swap_{len(swaps) + 1:04d}', residual, paths, swap_end)
            swaps[new.id] = new
            values[new.id] = bpv_swap(new, paths)
            coupon_cash[new.id] = 0.0
        net = risk['bpv'] + sum((v['bpv'] for v in values.values()))
        if abs(net) > 1e-06:
            raise ArithmeticError(f'BPV non neutralisée : {net} au {date}.')
        gross = sum((committed_notional_at(s, date) for s in swaps.values()))
        signed = sum(
            (
                (1 if s.pay_fixed else -1) * committed_notional_at(s, date) for s in swaps.values()
            )
        )
        for key, s in (swaps.items() if save_swaps else []):
            if save_swaps == 'transactions' and s.trade_date != date:
                continue
            if s.closed_date is not None and s.closed_date < date or s.end_date < date:
                continue
            audit.append(
                dict(
                    strategie=name,
                    date=date,
                    swap_id=key,
                    date_transaction=s.trade_date,
                    debut=s.start_date,
                    echeance=s.end_date,
                    notionnel_eur=s.notional,
                    sens='payeur_fixe' if s.pay_fixed else 'receveur_fixe',
                    taux_fixe_pct=s.fixed_rate * 100,
                    prix_mtm_eur=values[key]['pv'],
                    bpv_eur_par_pb=values[key]['bpv'],
                    coupon_net_paye_eur=coupon_cash[key],
                    soulte_unwind_eur=closed.get(key, 0.0),
                    ferme_ce_mois=key in closed
                )
            )
        if previous is not None:
            trace_index = old_trace.dates.get_loc(date)
            margin = (
                carry['loan_interest_eur'] - carry['funding_cost_eur']
                - carry['liquidity_cost_eur'] + carry['swap_cashflow_eur']
                + settlement + fee
            )
            cumulative += margin
            monthly.append(
                dict(
                    strategie=name,
                    date=date,
                    taux_pret_pct=old_loan.fixed_rate * 100,
                    itp_refinancement_pct=old_trace.reference_itp[0, trace_index] * 100,
                    taux_client_propose_pct=old_trace.offered_rates[0, trace_index] * 100,
                    spread_refinancement_bp=old_trace.spread_bps[0, trace_index],
                    proba_prepaiement_mois=old_trace.conditional_probability[0, trace_index],
                    proba_prepaiement_cumulee=old_trace.cumulative_probability[0, trace_index],
                    taux_financement_pct=float(central.loc[previous, 'taux_6m']) * 100,
                    encours_debut_eur=carry['opening_outstanding'],
                    interets_pret_eur=carry['loan_interest_eur'],
                    cout_financement_eur=carry['funding_cost_eur'],
                    cout_liquidite_eur=carry['liquidity_cost_eur'],
                    flux_swaps_eur=carry['swap_cashflow_eur'],
                    soulte_unwind_eur=settlement,
                    principal_prepaye_eur=principal,
                    indemnite_prepaiement_eur=fee,
                    marge_mois_eur=margin,
                    marge_cumulee_eur=cumulative,
                    bpv_nette_eur_par_pb=net,
                    notionnel_brut_eur=gross,
                    notionnel_net_signe_eur=signed,
                    echeance_swap_cible=swap_end
                )
            )
        previous = date
        clear_pricing_cache(paths)
    return (pd.DataFrame(monthly), pd.DataFrame(audit))
