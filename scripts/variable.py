"""Prêts révisables sans prépaiement et protection par swaptions bullets."""
from dataclasses import dataclass, replace
from math import erfc, exp, pi, sqrt

import numpy as np
import pandas as pd

from scripts.hw import parallel_bump, forward_rate_for_period
from scripts.model import (
    Loan, Swap, build_amortization_profile, fixed_cashflows, outstanding_at,
    payment_dates, build_bpv_hedge_swap, bpv_swap, update_known_fixings,
    swap_cash_income, committed_notional_at, clear_pricing_cache,
)


@dataclass(frozen=True)
class Swaption:
    id: str
    expiry: pd.Timestamp
    start: pd.Timestamp
    end: pd.Timestamp
    notional: float
    strike: float


def revision_dates(start, end, months):
    if type(months) is not int or months < 1:
        raise ValueError('La fréquence de révision doit être un nombre entier de mois.')
    dates = [pd.Timestamp(start)]
    while dates[-1] + pd.offsets.MonthEnd(months) < end:
        dates.append(dates[-1] + pd.offsets.MonthEnd(months))
    return pd.DatetimeIndex(dates + [pd.Timestamp(end)])


def revised_rate(offered, cap=None):
    """Appliquer chaque nouvelle référence, sans seuil, puis son plafond éventuel."""
    offered = np.asarray(offered, dtype=float)
    return offered if cap is None else np.minimum(offered, cap)


def bonds_at(paths, date, payments):
    """Prix HW P(t,T) par chemin, à l'état effectivement atteint à t."""
    date = pd.Timestamp(date)
    times = (pd.DatetimeIndex(payments) - date).days.to_numpy() / 365
    elapsed = (date - paths.dates[0]).days / 365
    column = paths.dates.get_loc(date)
    a, sigma = paths.parameters.a, paths.parameters.sigma
    b = -np.expm1(-a * times) / a
    state = paths.taux_court[:, column]
    curve = paths.courbe
    return curve.discount(elapsed + times)[None, :] / curve.discount(elapsed) * np.exp(
        -b[None, :] * (state - curve.forward_instantane(elapsed))[:, None]
        -sigma**2 / (4*a) * (-np.expm1(-2*a*elapsed)) * b[None, :]**2
    )


def block_itp(paths, date, end, cashflows):
    """ITP de la période fixe, avec l'encours résiduel au pair à sa fin."""
    cache = getattr(paths, '_variable_itp', None)
    if cache is None:
        cache = {}
        paths._variable_itp = cache
    key = (pd.Timestamp(date), pd.Timestamp(end))
    if key not in cache:
        block = cashflows.loc[(cashflows.index > date) & (cashflows.index <= end)]
        discounts = bonds_at(paths, date, block.index)
        opening = block.outstanding.to_numpy()
        principal = block.principal.to_numpy()
        residual = float(block.closing_outstanding.iloc[-1])
        numerator = opening[0] - discounts @ principal - residual * discounts[:, -1]
        cache[key] = numerator / (discounts @ (opening / 12))
    return cache[key]


def loan_value(paths, cashflows, base_rate, resets, cap, margin):
    """Valeur de tous les flux futurs, révisions contractuelles comprises."""
    values = np.zeros(len(paths.taux_6m))
    rate = np.full(len(values), base_rate)
    future = cashflows.loc[cashflows.index > paths.dates[0]]
    for date, payment in future.iterrows():
        column = paths.dates.get_loc(date)
        amount = payment.principal + payment.outstanding * (rate + margin) / 12
        values += amount * paths.facteurs_actualisation[:, column]
        if date in resets[:-1]:
            following = resets[resets.get_loc(date) + 1]
            offered = block_itp(paths, date, following, cashflows)
            rate = revised_rate(offered, cap)
    return float(values.mean())


def swap_terms(option, curve):
    dates = payment_dates(option.start, option.end)
    starts = pd.DatetimeIndex([option.start]).append(dates[:-1])
    times = (dates - curve.date).days.to_numpy() / 365
    delta = (dates - starts).days.to_numpy() / 365
    discounts = curve.discount(times)
    start_time = (option.start - curve.date).days / 365
    numerator = float(curve.discount(start_time) - discounts[-1])
    annuity = float(discounts @ delta)
    return numerator / annuity, annuity, times, delta, discounts


def swaption_quote(option, curve, hw, normal_vol=None):
    """Prix Bachelier et paramètres permettant de vérifier la prime."""
    expiry = (option.expiry - curve.date).days / 365
    if expiry < 0:
        return {'prix_eur': 0.}
    forward, annuity, times, delta, discounts = swap_terms(option, curve)
    if normal_vol is None:
        # Linéariser le taux de swap par rapport à l'état gaussien à l'exercice.
        a = hw.a
        start = (option.start - option.expiry).days / 365
        remaining = times - expiry
        b = -np.expm1(-a * remaining) / a
        b_start = -np.expm1(-a * start) / a
        p_start = curve.discount((option.start - curve.date).days / 365)
        derivative_n = -b_start * p_start + b[-1] * discounts[-1]
        derivative_a = -float((delta * b) @ discounts)
        sensitivity = (derivative_n - forward * derivative_a) / annuity
        std = abs(sensitivity) * hw.sigma * sqrt(-np.expm1(-2*a*expiry) / (2*a))
    else:
        if not np.isfinite(normal_vol) or normal_vol < 0:
            raise ValueError('La volatilité normale doit être positive ou nulle.')
        std = normal_vol * sqrt(expiry)
    intrinsic = forward - option.strike
    if std < 1e-14:
        price = option.notional * annuity * max(intrinsic, 0.)
    else:
        d = intrinsic / std
        cdf = .5 * erfc(-d / sqrt(2))
        density = exp(-.5*d*d) / sqrt(2*pi)
        price = max(0., option.notional * annuity * (intrinsic*cdf + std*density))
    return {
        'prix_eur': price, 'forward_initial_pct': forward*100,
        'strike_pct': option.strike*100, 'ecart_forward_strike_bp': intrinsic*10000,
        'ecart_type_taux_bp': std*10000,
        'vol_normale_implicite_bp': std/sqrt(expiry)*10000 if expiry>0 else 0.,
        'annuite_actualisee_ans': annuity, 'expiration_ans': expiry,
        'duree_swap_ans': (option.end-option.start).days/365,
    }



def swaption_price(option, curve, hw, normal_vol=None):
    return swaption_quote(option, curve, hw, normal_vol)['prix_eur']


def swaption_price_hw(option, curve, hw):
    """Contrôle indépendant : payer bullet européen exact dans HW à un facteur.

    Espérance des zéros-coupons tronqués sous la mesure forward d'expiration.
    Référence sans décalage de fixing, comme les swaps variables simplifiés.
    """
    expiry = (option.expiry-curve.date).days/365
    if expiry < 0:
        return 0.
    dates = payment_dates(option.start, option.end)
    starts = pd.DatetimeIndex([option.start]).append(dates[:-1])
    delta = (dates-starts).days.to_numpy()/365
    maturities = pd.DatetimeIndex([option.start]).append(dates)
    times = (maturities-curve.date).days.to_numpy()/365
    discounts = curve.discount(times)
    weights = np.r_[1., -option.strike*delta]
    weights[-1] -= 1.
    variance = hw.sigma**2 * (-np.expm1(-2*hw.a*expiry))/(2*hw.a)
    if variance < 1e-28:
        return option.notional*max(float(weights@discounts),0.)
    if option.strike < 0:
        raise ValueError('Le contrôle HW analytique suppose un strike positif ou nul.')
    std = sqrt(variance)
    b = -np.expm1(-hw.a*(times-expiry))/hw.a

    def payoff(z):
        bonds = discounts * np.exp(-b*std*z-.5*variance*b*b)
        return float(weights@bonds)

    low, high = -40., 40.
    if payoff(high) <= 0:
        return 0.
    if payoff(low) >= 0:
        return option.notional*max(float(weights@discounts),0.)
    for _ in range(100):
        middle = (low+high)/2
        if payoff(middle)>0:
            high = middle
        else:
            low = middle
    boundary = (low+high)/2
    tails = np.array([.5*erfc((boundary+value*std)/sqrt(2)) for value in b])
    return option.notional*max(float(weights@(discounts*tails)),0.)


def premium_spread(premium, curve, cashflows):
    times = (cashflows.index - curve.date).days.to_numpy() / 365
    annuity = float(curve.discount(times) @ (cashflows.outstanding.to_numpy() / 12))
    return premium / annuity


def variable_strategy(parameters, context, settings, *, protected=False, save_swaps=True):
    """Une révision tous les trois ans et un ajustement de BPV à chaque mois."""
    _, simulation = context
    central = simulation.scenario_central
    start, end = central.index[0], central.index[-1]
    if not np.isfinite(settings['hausse_max']) or settings['hausse_max'] < 0:
        raise ValueError('La hausse maximale doit être positive ou nulle.')
    resets = revision_dates(start, end, settings['revision_mois'])
    name = 'variable_protected' if protected else 'variable_normal'
    profile = build_amortization_profile(start, end, parameters['nominal'])
    loan = Loan(name, parameters['nominal'], start, end, profile, fixed_rate=0.)
    cashflows = fixed_cashflows(loan)
    initial_paths = simulation.scenarios_conditionnels[start.strftime('%Y%m')]
    initial_rate = float(block_itp(initial_paths, start, resets[1], cashflows)[0])
    cap = initial_rate + settings['hausse_max'] if protected else None
    options = {}
    if protected:
        for i, expiry in enumerate(resets[1:-1], start=1):
            options[f'option_{i}'] = Swaption(
                f'option_{i}', expiry, expiry, resets[i+1],
                outstanding_at(loan, expiry), cap,
            )
    normal_vol = settings['vol_normale']
    option_costs = {key: swaption_price(option, initial_paths.courbe, initial_paths.parameters, normal_vol)
                    for key, option in options.items()}
    initial_quotes = {}
    for key, option in options.items():
        quote = swaption_quote(option, initial_paths.courbe, initial_paths.parameters, normal_vol)
        quote.pop('prix_eur')
        # Le contrôle exact n'est pas utilisé pour majorer artificiellement la prime.
        if option.strike >= 0:
            quote['prix_hw_controle_eur'] = swaption_price_hw(option, initial_paths.courbe, initial_paths.parameters)
        initial_quotes[key] = quote
    premium = sum(option_costs.values())
    fee_rate = premium_spread(premium, initial_paths.courbe, cashflows)
    margin = parameters['marge_commerciale'] + parameters['marge_liquidite'] + fee_rate
    base_rate = initial_rate
    swaps, swap_audit, option_audit, monthly = {}, [], [], []
    cumulative = smoothed_cumulative = 0.
    previous = None

    for date in central.index:
        paths = simulation.scenarios_conditionnels.get(date.strftime('%Y%m'))
        update_known_fixings(swaps, date, simulation)
        paid_rate = base_rate + margin
        coupon_cash = {key: swap_cash_income(swap, previous, date, central) if previous is not None else 0.
                       for key, swap in swaps.items()}
        opening = interest = funding = liquidity = fee_income = 0.
        if previous is not None:
            opening = float(cashflows.loc[date, 'outstanding'])
            interest = opening * paid_rate / 12
            funding = opening * float(central.loc[previous, 'taux_6m']) / 12
            liquidity = opening * parameters['marge_liquidite'] / 12
            fee_income = opening * fee_rate / 12

        exercised = set()
        for key, option in list(options.items()):
            if date != option.expiry:
                continue
            forward = swap_terms(option, paths.courbe)[0]
            if forward > option.strike:
                swap_id = f'exercice_{key}'
                swaps[swap_id] = Swap(
                    swap_id, option.start, option.end, option.notional, option.strike,
                    True, date, fixing_lag_business_days=0,
                    floating_fixings={date: float(forward_rate_for_period(
                        paths, date, date, payment_dates(date, option.end)[0])[0])},
                )
                coupon_cash[swap_id] = 0.
                exercised.add(key)

        for key in list(options):
            if options[key].expiry == date:
                option = options.pop(key)
                option_audit.append(dict(
                    strategie=name, date=date, option_id=key, exercice=key in exercised,
                    expiration=True, prix_mtm_eur=0., bpv_eur_par_pb=0.,
                    prime_initiale_eur=option_costs[key], echeance=option.expiry,
                    fin_swap=option.end, notionnel_eur=option.notional, strike_pct=option.strike*100,
                ))

        if date in resets[:-1] and date != start:
            next_reset = resets[resets.get_loc(date) + 1]
            offered = float(block_itp(paths, date, next_reset, cashflows)[0])
            base_rate = float(revised_rate(offered, cap))
        next_reset = resets[resets > date][0] if date < end else end
        values, option_bpv, loan_bpv = {}, 0., 0.
        if paths is not None:
            bumped = parallel_bump(paths, 1e-4)
            for key, option in options.items():
                price = swaption_price(option, paths.courbe, paths.parameters, normal_vol)
                up = swaption_price(option, bumped.courbe, paths.parameters, normal_vol)
                option_bpv += up - price
                option_audit.append(dict(
                    strategie=name, date=date, option_id=key, exercice=False, expiration=False,
                    prix_mtm_eur=price, bpv_eur_par_pb=up-price,
                    prime_initiale_eur=option_costs[key], echeance=option.expiry,
                    fin_swap=option.end, notionnel_eur=option.notional, strike_pct=option.strike*100,
                ))
            loan_bpv = (loan_value(bumped, cashflows, base_rate, resets, cap, margin)
                        - loan_value(paths, cashflows, base_rate, resets, cap, margin))
            values = {key: bpv_swap(swap, paths) for key, swap in swaps.items()}
            residual = loan_bpv + option_bpv + sum(value['bpv'] for value in values.values())
            if abs(residual) > 1e-8:
                swap_id = f'ajustement_{date:%Y%m}'
                # Une seule mensualité restante : le swap spot au pair a deux
                # coupons connus et une BPV nulle. Garder alors le fixing futur.
                last_period = len(payment_dates(date, next_reset)) == 1
                swaps[swap_id] = build_bpv_hedge_swap(
                    swap_id, residual, paths, next_reset,
                    start_lag=4 if last_period else 0,
                    fixing_lag=2 if last_period else 0,
                )
                values[swap_id] = bpv_swap(swaps[swap_id], paths)
                coupon_cash[swap_id] = 0.
        net = loan_bpv + option_bpv + sum(value['bpv'] for value in values.values())
        if abs(net) > 1e-6:
            raise ArithmeticError(f'BPV variable non neutralisée au {date}: {net}')

        if save_swaps:
            for key, swap in swaps.items():
                if save_swaps == 'transactions' and swap.trade_date != date:
                    continue
                if swap.end_date < date:
                    continue
                value = values.get(key, {'pv': 0., 'bpv': 0.})
                swap_audit.append(dict(
                    strategie=name, date=date, swap_id=key, date_transaction=swap.trade_date,
                    debut=swap.start_date, echeance=swap.end_date, notionnel_eur=swap.notional,
                    sens='payeur_fixe' if swap.pay_fixed else 'receveur_fixe',
                    taux_fixe_pct=swap.fixed_rate*100, prix_mtm_eur=value['pv'],
                    bpv_eur_par_pb=value['bpv'], coupon_net_paye_eur=coupon_cash[key],
                    soulte_unwind_eur=0., ferme_ce_mois=False,
                    origine='exercice_swaption' if key.startswith('exercice') else 'ajustement',
                    convention='depart_differe_dernier_mois' if swap.start_date>swap.trade_date else 'depart_immediat',
                ))
        premium_paid = premium if date == start else 0.
        cash_margin = interest - funding - liquidity + sum(coupon_cash.values()) - premium_paid
        smoothed_margin = cash_margin + premium_paid - fee_income
        cumulative += cash_margin
        smoothed_cumulative += smoothed_margin
        monthly.append(dict(
            strategie=name, date=date, taux_pret_pct=paid_rate*100,
            taux_apres_revision_pct=(base_rate+margin)*100,
            taux_financement_pct=float(central.loc[previous if previous is not None else date, 'taux_6m'])*100,
            encours_debut_eur=opening, interets_pret_eur=interest, cout_financement_eur=funding,
            cout_liquidite_eur=liquidity, flux_swaps_eur=sum(coupon_cash.values()),
            flux_swaps_exerces_eur=sum(value for key,value in coupon_cash.items() if key.startswith('exercice')),
            flux_swaps_ajustement_eur=sum(value for key,value in coupon_cash.items() if not key.startswith('exercice')),
            nombre_swaptions_exercees=len(exercised),
            soulte_unwind_eur=0., principal_prepaye_eur=0., indemnite_prepaiement_eur=0.,
            marge_mois_eur=cash_margin, marge_cumulee_eur=cumulative,
            bpv_nette_eur_par_pb=net, bpv_pret_eur_par_pb=loan_bpv,
            bpv_options_eur_par_pb=option_bpv,
            notionnel_brut_eur=sum(committed_notional_at(swap,date) for swap in swaps.values()),
            notionnel_net_signe_eur=sum((1 if swap.pay_fixed else -1)*committed_notional_at(swap,date) for swap in swaps.values()),
            echeance_swap_cible=next_reset, revision_ce_mois=date in resets[1:-1],
            prime_swaptions_payee_eur=premium_paid, cout_protection_lisse_eur=fee_income,
            supplement_client_protection_bp=fee_rate*10000,
            marge_lissee_eur=smoothed_margin, marge_cumulee_lissee_eur=smoothed_cumulative,
            taux_6m_central_pct=float(central.loc[date,'taux_6m'])*100,
        ))
        previous = date
        clear_pricing_cache(paths)
    for row in option_audit:
        if row['date'] == start:
            row.update(initial_quotes[row['option_id']])
    return pd.DataFrame(monthly), pd.DataFrame(swap_audit), pd.DataFrame(option_audit)
