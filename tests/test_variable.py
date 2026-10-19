import contextlib
from dataclasses import replace
import io
from math import sqrt, pi
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import config
from scripts.hw import HWParameters, courbe_initiale
from scripts.model import market, fixed_cashflows, Loan, build_amortization_profile
from scripts.variable import (
    Swaption, revised_rate, swaption_price, swap_terms, premium_spread,
    variable_strategy, block_itp, bonds_at,
)


class VariableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parameters = dict(config.PARAMETRES, maturite_ans=4, seed=2,
                              profil_initial='montante', hw_a=.1, hw_sigma=0.,
                              chemins_conditionnels=2)
        cls.context = market(cls.parameters)
        cls.settings = dict(config.VARIABLE, hausse_max=0.)
        cls.normal = variable_strategy(cls.parameters, cls.context, cls.settings, protected=False)
        cls.protected = variable_strategy(cls.parameters, cls.context, cls.settings, protected=True)

    def test_every_reference_change_is_applied_without_threshold(self):
        self.assertAlmostEqual(float(revised_rate(.024)),.024)
        self.assertEqual(float(revised_rate(.025000000001)),.025000000001)
        self.assertAlmostEqual(float(revised_rate(.03,.025)),.025)
        np.testing.assert_allclose(revised_rate([.024,.0262]),[.024,.0262])

    def test_bachelier_price_at_the_money(self):
        curve = courbe_initiale('2025-01-31','plate')
        option = Swaption('test',pd.Timestamp('2028-01-31'),pd.Timestamp('2028-02-04'),
                          pd.Timestamp('2031-01-31'),100000.,0.)
        forward, annuity, *_ = swap_terms(option,curve)
        option = replace(option,strike=forward)
        expiry = (option.expiry-curve.date).days/365
        expected = option.notional*annuity*.005*sqrt(expiry)/sqrt(2*pi)
        self.assertAlmostEqual(swaption_price(option,curve,HWParameters(.1,.005),.005),expected,places=8)
        self.assertEqual(swaption_price(option,curve,HWParameters(.1,0.),0.),0.)

    def test_block_itp_values_amortization_and_terminal_balance_at_par(self):
        _, simulation = self.context
        start = simulation.scenario_central.index[0]
        end = simulation.scenario_central.index[-1]
        paths = simulation.scenarios_conditionnels[start.strftime('%Y%m')]
        loan = Loan('test',100000.,start,end,build_amortization_profile(start,end,100000.),fixed_rate=0.)
        cashflows = fixed_cashflows(loan)
        block_end = start+pd.offsets.MonthEnd(36)
        block = cashflows.loc[cashflows.index<=block_end]
        rate = block_itp(paths,start,block_end,cashflows)
        discount = bonds_at(paths,start,block.index)
        pv = discount @ block.principal.to_numpy()+rate*(discount@(block.outstanding.to_numpy()/12))
        pv += discount[:,-1]*block.closing_outstanding.iloc[-1]
        np.testing.assert_allclose(pv,100000.,atol=1e-7)

    def test_no_prepayment_and_monthly_bpv_target(self):
        for data, swaps, options in [self.normal,self.protected]:
            self.assertEqual(len(data),49)
            self.assertEqual(data.principal_prepaye_eur.sum(),0.)
            self.assertEqual(data.soulte_unwind_eur.sum(),0.)
            np.testing.assert_allclose(data.bpv_nette_eur_par_pb,0.,atol=1e-6)
            for _, trade in swaps.loc[swaps.date==swaps.date_transaction].iterrows():
                expected = pd.Timestamp('2028-01-31') if trade.date_transaction<pd.Timestamp('2028-01-31') else pd.Timestamp('2029-01-31')
                self.assertEqual(trade.echeance,expected)
            np.testing.assert_allclose(data.encours_debut_eur.iloc[1:],100000.*(48-np.arange(48))/48,atol=1e-7)

    def test_old_coupon_is_paid_before_revision_and_new_coupon_after(self):
        data = self.normal[0].set_index('date')
        revision = pd.Timestamp('2028-01-31')
        old = data.loc[revision-pd.offsets.MonthEnd(1),'taux_pret_pct']
        self.assertAlmostEqual(data.loc[revision,'taux_pret_pct'],old)
        new = data.loc[revision,'taux_apres_revision_pct']
        self.assertNotAlmostEqual(new,old)
        self.assertAlmostEqual(data.loc[revision+pd.offsets.MonthEnd(1),'taux_pret_pct'],new)
        self.assertEqual(data.revision_ce_mois.sum(),1)

    def test_premium_is_recovered_in_client_rate_and_not_paid_twice(self):
        data, swaps, options = self.protected
        premium = data.prime_swaptions_payee_eur.sum()
        self.assertGreater(premium,0.)
        self.assertEqual((data.prime_swaptions_payee_eur>0).sum(),1)
        self.assertAlmostEqual(data.marge_mois_eur.iloc[0],-premium)
        curve = self.context[1].central_paths.courbe
        elapsed = (pd.DatetimeIndex(data.date)-curve.date).days.to_numpy()/365
        recovered = np.dot(curve.discount(elapsed),data.cout_protection_lisse_eur)
        self.assertAlmostEqual(recovered,premium,places=7)
        expected = data.interets_pret_eur-data.cout_financement_eur-data.cout_liquidite_eur+data.flux_swaps_eur-data.prime_swaptions_payee_eur
        np.testing.assert_allclose(data.marge_mois_eur,expected,atol=1e-9)
        np.testing.assert_allclose(data.marge_cumulee_eur,expected.cumsum(),atol=1e-9)
        self.assertEqual(data.prime_swaptions_payee_eur.iloc[36],0.)
        self.assertTrue(options.exercice.any())
        self.assertTrue((swaps.origine=='exercice_swaption').any())
        limit = data.taux_pret_pct.iloc[0]
        self.assertLessEqual(data.taux_apres_revision_pct.max(),limit+1e-9)

    def test_normal_and_protected_coincide_when_protection_is_free_and_never_used(self):
        settings = dict(self.settings,hausse_max=1.)
        protected, _, _ = variable_strategy(self.parameters,self.context,settings,protected=True)
        self.assertEqual(protected.prime_swaptions_payee_eur.sum(),0.)
        np.testing.assert_allclose(protected.marge_mois_eur,self.normal[0].marge_mois_eur,atol=1e-8)

    def test_future_swaptions_are_in_the_monthly_bpv_target(self):
        data, swaps, options = self.protected
        self.assertGreater(data.bpv_options_eur_par_pb.abs().max(), .001)
        swap_risk = swaps.groupby('date').bpv_eur_par_pb.sum().reindex(data.date).fillna(0).to_numpy()
        total = data.bpv_pret_eur_par_pb + data.bpv_options_eur_par_pb + swap_risk
        np.testing.assert_allclose(total, 0., atol=1e-6)

    def test_transactions_mode_preserves_margins_and_stores_only_trades(self):
        data, swaps, _ = variable_strategy(self.parameters,self.context,self.settings,
                                          protected=True,save_swaps='transactions')
        np.testing.assert_allclose(data.marge_mois_eur,self.protected[0].marge_mois_eur,atol=1e-8)
        self.assertTrue((swaps.date==swaps.date_transaction).all())
        self.assertFalse(swaps.duplicated(['strategie','swap_id']).any())

    def test_full_pipeline_saves_six_strategies_and_plots_one_seed(self):
        import run
        from plot import plot_batch
        experiment = dict(profil='montante',a=.1,sigma=0.,maturite_ans=4,seed=2)
        with patch.object(config,'PRODUITS',['fixe','variable','protected']), \
             patch.dict(config.PARAMETRES,chemins_conditionnels=2), \
             patch.dict(config.VARIABLE,self.settings), \
             patch.object(config,'SAUVER_SWAPS',True), \
             patch.dict(config.PREPAIEMENT,intercept=-700,beta_change=0,beta_gain=0):
            tables = run.run_experiment(experiment)
            self.assertEqual(tables['marges'].strategie.nunique(),6)
            self.assertFalse(tables['swaptions'].empty)
            with tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                database = run.open_database(folder/'batch.sqlite')
                run.save_experiment(database,experiment,tables)
                run.export_csv(database,folder)
                database.close()
                with contextlib.redirect_stdout(io.StringIO()):
                    plot_batch(folder,seed=2)
                self.assertTrue((folder/'swaptions.csv').exists())
                self.assertEqual(len(list((folder/'figures_seed_2').rglob('scenario_central.png'))),1)
                self.assertEqual(len(list((folder/'figures_seed_2').rglob('swaps_pris.csv'))),1)
                with self.assertRaises(ValueError):
                    plot_batch(folder,seed=99)

    def test_database_accepts_new_columns_without_moving_old_values(self):
        import run
        database = run.open_database(':memory:')
        with database:
            run.insert_table(database,'sample',pd.DataFrame({'experience':['old'],'amount':[3.]}))
            run.insert_table(database,'sample',pd.DataFrame({'amount':[4.],'experience':['new'],'premium':[2.]}))
        old = database.execute("SELECT amount,premium FROM sample WHERE experience='old'").fetchone()
        new = database.execute("SELECT amount,premium FROM sample WHERE experience='new'").fetchone()
        self.assertEqual(old,(3.,None))
        self.assertEqual(new,(4.,2.))
        database.close()


class RevisionRegressionTests(unittest.TestCase):
    def test_small_old_premiums_come_from_the_old_out_of_money_strike(self):
        from scripts.variable import swaption_quote, swaption_price_hw
        curve = courbe_initiale('2025-01-31','plate')
        hw = HWParameters(.05,.003)
        old_prices = [.000870087779586024,.08596203606555772,.29683611077801214,.3258742701383374]
        new_sum = reference_sum = 0.
        for i, old_price in enumerate(old_prices, start=1):
            expiry = pd.Timestamp('2025-01-31')+pd.offsets.MonthEnd(36*i)
            end = expiry+pd.offsets.MonthEnd(36)
            old = Swaption(str(i),expiry,expiry+pd.offsets.BDay(4),end,100000*(1-i/5),.04501670392472408)
            self.assertAlmostEqual(swaption_price(old,curve,hw),old_price,places=9)
            new = replace(old,start=expiry,strike=old.strike-.02)
            quote = swaption_quote(new,curve,hw)
            exact = swaption_price_hw(new,curve,hw)
            self.assertLess(abs(quote['prix_eur']-exact),.02)
            self.assertLess(abs(quote['ecart_forward_strike_bp']),1.)
            new_sum += quote['prix_eur']
            reference_sum += exact
        self.assertGreater(new_sum,1000.)
        self.assertLess(abs(new_sum-reference_sum),.1)

    def test_exact_hw_price_matches_independent_gaussian_quadrature(self):
        from scripts.variable import swaption_price_hw
        from scripts.model import payment_dates
        curve = courbe_initiale('2025-01-31','plate')
        hw = HWParameters(.05,.003)
        option = Swaption('test',pd.Timestamp('2028-01-31'),pd.Timestamp('2028-01-31'),
                          pd.Timestamp('2031-01-31'),80000.,.025)
        dates = payment_dates(option.start,option.end)
        starts = pd.DatetimeIndex([option.start]).append(dates[:-1])
        maturities = pd.DatetimeIndex([option.start]).append(dates)
        times = (maturities-curve.date).days.to_numpy()/365
        expiry = (option.expiry-curve.date).days/365
        variance = hw.sigma**2*(-np.expm1(-2*hw.a*expiry))/(2*hw.a)
        b = -np.expm1(-hw.a*(times-expiry))/hw.a
        weights = np.r_[1.,-option.strike*(dates-starts).days.to_numpy()/365]
        weights[-1] -= 1
        nodes, gaussian_weights = np.polynomial.hermite.hermgauss(256)
        bonds = curve.discount(times)[None,:]*np.exp(-b[None,:]*sqrt(2*variance)*nodes[:,None]-.5*variance*b[None,:]**2)
        payoffs = np.maximum(bonds@weights,0.)
        numerical = option.notional*float(gaussian_weights@payoffs)/sqrt(pi)
        exact = swaption_price_hw(option,curve,hw)
        self.assertLess(abs(numerical-exact)/exact,.005)

    def test_zero_lag_coupon_fixing_on_a_weekend_and_full_first_payment(self):
        from scripts.model import build_bpv_hedge_swap, fixing_schedule
        from scripts.hw import forward_rate_for_period
        p = dict(config.PARAMETRES,maturite_ans=4,seed=2,profil_initial='montante',
                 hw_a=.1,hw_sigma=0.,chemins_conditionnels=2)
        context = market(p)
        date = pd.Timestamp('2026-02-28')  # samedi
        paths = context[1].scenarios_conditionnels[date.strftime('%Y%m')]
        hedge = build_bpv_hedge_swap('test',-10.,paths,pd.Timestamp('2028-01-31'),start_lag=0,fixing_lag=0)
        self.assertEqual(hedge.start_date,date)
        self.assertIn(date,hedge.floating_fixings)
        starts, ends, fixings = fixing_schedule(hedge.start_date,hedge.end_date,0)
        np.testing.assert_array_equal(starts,fixings)
        data, trades, _ = variable_strategy(p,context,dict(config.VARIABLE,hausse_max=0.),protected=True,save_swaps='transactions')
        exercise = trades.loc[trades.origine=='exercice_swaption'].iloc[0]
        self.assertEqual(exercise.debut,exercise.date_transaction)
        first_payment = exercise.debut+pd.offsets.MonthEnd(1)
        days = (first_payment-exercise.debut).days
        fixing = float(forward_rate_for_period(context[1].central_paths,exercise.debut,exercise.debut,first_payment)[0])
        expected = exercise.notionnel_eur*days*(fixing/360-exercise.taux_fixe_pct/100/365)
        actual = data.loc[data.date==first_payment,'flux_swaps_exerces_eur'].iloc[0]
        self.assertAlmostEqual(actual,expected,places=8)
        deferred = trades.loc[trades.debut>trades.date_transaction]
        self.assertFalse(deferred.empty)
        self.assertTrue((deferred.convention=='depart_differe_dernier_mois').all())
        for _, row in deferred.iterrows():
            self.assertEqual(len(pd.date_range(row.date_transaction+pd.offsets.MonthEnd(1),row.echeance,freq='ME')),1)
        reset_trades = trades.loc[trades.date_transaction==pd.Timestamp('2028-01-31')]
        self.assertTrue((reset_trades.debut==reset_trades.date_transaction).all())

    def test_variable_tariff_has_no_prepayment_or_refinancing_premium(self):
        import scripts.model as model
        p = dict(config.PARAMETRES,maturite_ans=1,seed=2,profil_initial='plate',
                 hw_a=.1,hw_sigma=.003,chemins_conditionnels=2)
        context = market(p)
        before = variable_strategy(p,context,config.VARIABLE,protected=False,save_swaps=False)[0]
        with patch.object(model,'add_prepayment',side_effect=AssertionError('Prépaie­ment interdit ici')):
            after = variable_strategy(dict(p,marge_refinancement=.50),context,config.VARIABLE,protected=False,save_swaps=False)[0]
        np.testing.assert_allclose(before.taux_pret_pct,after.taux_pret_pct)
        np.testing.assert_allclose(before.marge_mois_eur,after.marge_mois_eur)
