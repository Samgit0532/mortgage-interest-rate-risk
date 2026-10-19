import unittest
from pathlib import Path
import numpy as np
from scripts.model import market,strategy,STRATEGIES
from scripts.model import PrepaymentSettings


class MinimalSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from config import PARAMETRES
        cls.parameters={**PARAMETRES,'seed':1,'hw_a':.1,'hw_sigma':.005,'profil_initial':'plate','maturite_ans':1,'chemins_conditionnels':2}
        cls.context=market(cls.parameters)

    def check(self,settings,parameters=None):
        results={}
        for name,expected,unwind in STRATEGIES:
            d,s=strategy(self.parameters if parameters is None else parameters,self.context,name,expected,unwind,prepayment_settings=settings)
            self.assertEqual(len(d),12)
            np.testing.assert_allclose(d.bpv_nette_eur_par_pb,0.,atol=1e-6)
            expected_cash=d.interets_pret_eur-d.cout_financement_eur-d.cout_liquidite_eur+d.flux_swaps_eur+d.soulte_unwind_eur+d.indemnite_prepaiement_eur
            np.testing.assert_allclose(d.marge_mois_eur,expected_cash,atol=1e-9)
            np.testing.assert_allclose(d.marge_cumulee_eur,expected_cash.cumsum(),atol=1e-9)
            self.assertEqual(d.notionnel_brut_eur.iloc[-1],0.)
            diagnostics=['itp_refinancement_pct','taux_client_propose_pct','spread_refinancement_bp']
            self.assertTrue(np.isfinite(d.select_dtypes('number').drop(columns=diagnostics)).all().all())
            self.assertTrue(np.isfinite(d[diagnostics].iloc[:-1]).all().all())
            np.testing.assert_allclose(d.encours_debut_eur,self.parameters['nominal']*(12-np.arange(12))/12,atol=1e-7)
            np.testing.assert_allclose(s[s.date==s.date_transaction].prix_mtm_eur,0.,atol=1e-6)
            if not unwind:
                self.assertFalse(s.ferme_ce_mois.any());self.assertEqual(d.soulte_unwind_eur.abs().sum(),0.)
            results[name]=(d,s)
        return results

    def test_without_prepayment_same_results_for_unwind_and_conservation(self):
        results=self.check(PrepaymentSettings(-700,0,0,0))
        for prefix in ['fin_contractuelle','fin_esperee']:
            np.testing.assert_allclose(results[prefix+'_unwind'][0].marge_mois_eur,results[prefix+'_conservation'][0].marge_mois_eur)
        self.assertTrue(all(d.principal_prepaye_eur.sum()==0. for d,s in results.values()))

    def test_with_prepayment_retention_and_unwind_are_distinct(self):
        # Un résiduel élevé permet aussi un prépaiement sans gain de taux.
        results=self.check(PrepaymentSettings(-1,0,0,0))
        for prefix in ['fin_contractuelle','fin_esperee']:
            a,book=results[prefix+'_unwind'];b,retained=results[prefix+'_conservation']
            self.assertTrue(book.ferme_ce_mois.any())
            self.assertGreater(a.principal_prepaye_eur.sum(),0.)
            np.testing.assert_allclose(a.principal_prepaye_eur,b.principal_prepaye_eur)
            np.testing.assert_allclose(a.interets_pret_eur,b.interets_pret_eur)
            self.assertGreater(np.max(np.abs(a.marge_mois_eur-b.marge_mois_eur)),1e-5)
            np.testing.assert_allclose(book.groupby('date').soulte_unwind_eur.sum().reindex(a.date).to_numpy(),a.soulte_unwind_eur,atol=1e-8)

    def test_residual_prepayment_even_when_offered_tariff_is_higher(self):
        import pandas as pd
        from scripts.model import Loan
        from scripts.model import build_amortization_profile
        from scripts.model import add_prepayment
        start=pd.Timestamp('2025-01-31');end=pd.Timestamp('2025-06-30')
        loan=Loan('test',100000.,start,end,build_amortization_profile(start,end,100000.,.01),fixed_rate=.01)
        dates=pd.date_range(start,end,freq='ME')
        rates=pd.DataFrame({'rate_6m':[.03,.031,.03,.02,.035,.03]},index=dates)
        history=pd.Series(.03,index=pd.date_range('2024-06-30','2024-12-31',freq='ME'))
        settings=PrepaymentSettings(-2,0,0,0,liquidity_margin=0,product_margin=0,
            refinancing_curve=self.context[0].courbe('202501'))
        result=add_prepayment(rates,loan,history,settings)
        expected_probability=1-(1-1/(1+np.exp(2)))**3
        np.testing.assert_allclose(result.conditional_probability[0],[0,expected_probability,expected_probability,expected_probability,expected_probability,0])
        self.assertTrue((result.spread_bps[0,1:-1]<0).all())
        self.assertGreater(result.cumulative_probability[0,2],result.cumulative_probability[0,1])
        self.assertTrue(result.prepay.any())
        np.testing.assert_allclose(result.cumulative_probability,1-np.cumprod(1-result.conditional_probability,axis=1))
        from dataclasses import replace
        penalized=add_prepayment(rates,loan,history,replace(settings,beta_gain=.01))
        remaining=(dates[1:-1]-end).days.to_numpy()/-365
        score=-2+.01*penalized.spread_bps[0,1:-1]*remaining
        logistic=1/(1+np.exp(-score))
        np.testing.assert_allclose(penalized.conditional_probability[0,1:-1],1-(1-logistic)**3)
        self.assertTrue((penalized.conditional_probability[0,1:-1]<expected_probability).all())
        tiny=add_prepayment(rates,loan,history,replace(settings,intercept=-80))
        self.assertTrue((tiny.conditional_probability[0,1:-1]>0).all())
        self.assertGreater(tiny.cumulative_probability[0,-1],0.)

    def test_itp_refinancing_tariff_matches_actual_offered_loan(self):
        import pandas as pd
        from scripts.model import refinancing_tariffs,build_loan_with_itp
        repo,sim=self.context;curve=repo.courbe(self.parameters['debut']);end=sim.scenario_central.index[-1]
        dates=pd.DatetimeIndex(['2025-02-28','2025-06-30'])
        rates=np.array([[.01,.02],[.035,.04]])
        actual=refinancing_tariffs(dates,rates,end,curve,.011,.005)
        for i in range(2):
            for j,date in enumerate(dates):
                loan=build_loan_with_itp('new',100000.,date,end,curve.ancrer_sur_6m(date,rates[i,j]),
                    commercial_margin=.002,liquidity_margin=.005,refinancing_spread=.004,minimum_rate=.005)
                self.assertAlmostEqual(actual[i,j],loan.fixed_rate,places=12)

    def test_linear_amortization_and_itp_discounted_cashflows_at_par(self):
        import pandas as pd
        from scripts.model import Loan
        from scripts.model import build_amortization_profile,fixed_cashflows
        from scripts.model import calculate_itp
        repo,sim=self.context;curve=repo.courbe('202501');start=sim.scenario_central.index[0]
        for years in [1,15,25]:
            end=start+pd.DateOffset(years=years)
            profile=build_amortization_profile(start,end,100000.)
            itp=calculate_itp(curve,profile)
            loan=Loan('test',100000.,start,end,profile,fixed_rate=itp['annual_nominal_itp'])
            cf=fixed_cashflows(loan);discount=curve.discount((cf.index-start).days.to_numpy()/365)
            np.testing.assert_allclose(cf.principal,100000./(12*years),atol=1e-7)
            self.assertAlmostEqual(cf.principal.sum(),100000.,places=7)
            self.assertTrue((np.diff(cf.cashflow)<0).all())
            self.assertAlmostEqual(float(np.dot(cf.cashflow,discount)),100000.,places=7)
            self.assertAlmostEqual(itp['pv_at_itp'],100000.,places=7)

    def test_hw_parameters_held_at_every_projection_origin(self):
        repo,simulation=self.context
        self.assertEqual(repo.settings.hw_mode,'fixe')
        np.testing.assert_allclose(simulation.parametres.a,self.parameters['hw_a'])
        np.testing.assert_allclose(simulation.parametres.sigma,self.parameters['hw_sigma'])

    def test_hw_study_mode_produces_both_rate_directions(self):
        from scripts.hw import HWParameters,generer_chemins
        repo,simulation=self.context;start=simulation.scenario_central.index[0]
        import pandas as pd
        end=start+pd.DateOffset(years=15)
        curve=repo.courbe('202501')
        paths=generer_chemins(curve,end,HWParameters(.10,.005),256,np.random.default_rng(17))
        change=paths.taux_6m[:,-1]-paths.taux_6m[:,0]
        self.assertTrue((change<-.004).any())
        self.assertTrue((change>.004).any())

    def test_three_profiles_same_shocks_and_conditional_state(self):
        from scripts.hw import PROFILS
        from scripts.hw import generer_chemins
        contexts={p:market(dict(self.parameters,profil_initial=p)) for p in PROFILS}
        reference=None
        for profil,(repo,sim) in contexts.items():
            original=sim.central_paths
            times=(original.dates-original.dates[0]).days.to_numpy()/365.
            curve=repo.initial;a=repo.settings.hw_a;sigma=repo.settings.hw_sigma
            b=-np.expm1(-a*times)/a
            noise=original.taux_court[0]-curve.forward_instantane(times)-.5*sigma*sigma*b*b
            if reference is not None:
                np.testing.assert_allclose(noise,reference,atol=1e-14)
            reference=noise
            repeated=market(dict(self.parameters,profil_initial=profil))[1]
            np.testing.assert_array_equal(sim.scenario_central.taux_6m,repeated.scenario_central.taux_6m)
            for key,paths in sim.scenarios_conditionnels.items():
                date=paths.dates[0]
                np.testing.assert_allclose(paths.taux_6m[:,0],sim.scenario_central.loc[date,'taux_6m'],atol=1e-12)
                t=(date-curve.date).days/365.;u=.5;b=-np.expm1(-a*u)/a
                i=original.dates.get_loc(date)
                exact=curve.discount(t+u)/curve.discount(t)*np.exp(-b*(original.taux_court[0,i]-curve.forward_instantane(t))-sigma*sigma/(4*a)*(-np.expm1(-2*a*t))*b*b)
                self.assertAlmostEqual(float(paths.courbe.discount(u)),float(exact),places=13)
            changed=market(dict(self.parameters,profil_initial=profil,seed=self.parameters['seed']+1))[1]
            self.assertFalse(np.allclose(sim.scenario_central.taux_6m.iloc[1:],changed.scenario_central.taux_6m.iloc[1:]))

    def test_synthetic_itp_tariff_for_every_profile(self):
        from scripts.hw import courbe_initiale,PROFILS
        from scripts.model import refinancing_tariffs,build_loan_with_itp
        import pandas as pd
        dates=pd.DatetimeIndex(['2025-02-28','2025-06-30'])
        end=pd.Timestamp('2026-01-31');rates=np.array([[.02,.03]])
        for profil in PROFILS:
            curve=courbe_initiale('2025-01-31',profil)
            offered=refinancing_tariffs(dates,rates,end,curve,.011,.002491)
            for j,date in enumerate(dates):
                loan=build_loan_with_itp('test',100000.,date,end,curve.ancrer_sur_6m(date,rates[0,j]),commercial_margin=.002,liquidity_margin=.005,refinancing_spread=.004,minimum_rate=.002491)
                self.assertAlmostEqual(loan.fixed_rate,offered[0,j],places=12)



class BatchTests(unittest.TestCase):
    def setUp(self):
        from unittest.mock import patch
        import config
        selected = patch.object(config, 'PRODUITS', ['fixe'])
        selected.start()
        self.addCleanup(selected.stop)
        details = patch.object(config, 'SAUVER_SWAPS', False)
        details.start()
        self.addCleanup(details.stop)

    def test_almin_is_subtracted_after_the_cube(self):
        import pandas as pd
        from dataclasses import replace
        from scripts.model import Loan, add_prepayment, build_amortization_profile
        from scripts.hw import courbe_initiale
        start, end = pd.Timestamp('2025-01-31'), pd.Timestamp('2025-05-31')
        dates = pd.date_range(start, end, freq='ME')
        loan = Loan('test', 100000., start, end,
                    build_amortization_profile(start, end, 100000.), fixed_rate=.03)
        rates = pd.DataFrame({'rate_6m': .025}, index=dates)
        history = pd.Series(.025, index=pd.date_range('2024-07-31', '2024-12-31', freq='ME'))
        settings = PrepaymentSettings(-2, 0, 0, 0, refinancing_curve=courbe_initiale(start))
        base = add_prepayment(rates, loan, history, settings)
        for almin in [0., .02, .5, 1.]:
            result = add_prepayment(rates, loan, history, replace(settings, almin=almin))
            expected = np.maximum(base.conditional_probability - almin, 0.)
            np.testing.assert_allclose(result.conditional_probability, expected)
            np.testing.assert_allclose(result.cumulative_probability, 1-np.cumprod(1-expected, axis=1))
        with self.assertRaises(ValueError):
            add_prepayment(rates, loan, history, replace(settings, almin=-.01))

    def test_saving_is_atomic_and_idempotent(self):
        import pandas as pd
        from unittest.mock import patch
        import run
        experiment = dict(profil='plate', a=.1, sigma=.005, maturite_ans=1, seed=1)
        table = pd.DataFrame({'experience': [run.experience_key(experiment)], 'marge_mois_eur': [12.]})
        tables = {'marges': table, 'resume': table.copy()}
        database = run.open_database(':memory:')
        original = run.insert_table
        def fail_on_summary(connection, name, frame):
            if name == 'resume':
                raise RuntimeError('Interruption pendant la sauvegarde')
            original(connection, name, frame)
        with patch.object(run, 'insert_table', side_effect=fail_on_summary):
            with self.assertRaises(RuntimeError):
                run.save_experiment(database, experiment, tables)
        self.assertEqual(database.execute('SELECT count(*) FROM experiences').fetchone()[0], 0)
        self.assertIsNone(database.execute("SELECT name FROM sqlite_master WHERE name='marges'").fetchone())
        run.save_experiment(database, experiment, tables)
        run.save_experiment(database, experiment, tables)
        self.assertEqual(database.execute('SELECT count(*) FROM marges').fetchone()[0], 1)
        database.close()

    def test_batch_resume_csv_and_plots(self):
        import contextlib
        import io
        import tempfile
        import pandas as pd
        from unittest.mock import patch
        import config
        import run
        from plot import plot_batch, paired_differences
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with patch.dict(config.PARAMETRES, chemins_conditionnels=2), \
                 patch.object(config, 'MATURITES', [1]), \
                 patch.object(config, 'VOLATILITES', [.005]), \
                 patch.object(config, 'SEEDS', [1, 2]), \
                 patch.dict(config.PROFILS, {'plate': (.025, .025)}, clear=True), \
                 patch.object(run, 'run_experiment', wraps=run.run_experiment) as execute, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run.run_batch(folder, limit=1), 0)
                self.assertEqual(run.run_batch(folder, limit=1), 0)
                self.assertEqual(run.run_batch(folder), 0)
                self.assertEqual(execute.call_count, 2)
                plot_batch(folder, seed=1)
            monthly = pd.read_csv(folder/'marges.csv')
            self.assertEqual(len(monthly), 2*4*12)
            self.assertFalse(monthly.duplicated(['experience', 'strategie', 'date']).any())
            self.assertEqual(len(pd.read_csv(folder/'resume.csv')), 8)
            self.assertEqual(set(pd.read_csv(folder/'experiences.csv').statut), {'terminee'})
            self.assertFalse((folder/'swaps.csv').exists())
            self.assertEqual(len(list((folder/'figures_seed_1').rglob('*.png'))), 3)
            differences = paired_differences(monthly, 'fin_contractuelle_conservation')
            reference = differences[differences.strategie=='fin_contractuelle_conservation']
            np.testing.assert_allclose(reference.ecart_cumule_eur, 0.)

    def test_disabling_swap_details_does_not_change_margins(self):
        from scripts.model import market, strategy
        from config import PARAMETRES
        parameters = dict(PARAMETRES, seed=3, profil_initial='montante',
                          hw_a=.1, hw_sigma=.005, maturite_ans=1, chemins_conditionnels=2)
        settings = PrepaymentSettings(-1, 0, 0, 0, almin=.01)
        with_audit, audit = strategy(parameters, market(parameters), 'test', True, False,
                                    prepayment_settings=settings, save_swaps=True)
        without_audit, empty = strategy(parameters, market(parameters), 'test', True, False,
                                       prepayment_settings=settings, save_swaps=False)
        np.testing.assert_allclose(with_audit.marge_mois_eur, without_audit.marge_mois_eur)
        self.assertFalse(audit.empty)
        self.assertTrue(empty.empty)

    def test_batch_can_store_swap_prices_and_signed_unwinds(self):
        import tempfile
        import pandas as pd
        from unittest.mock import patch
        import config
        import run
        experiment = dict(profil='descendante', a=.1, sigma=.005, maturite_ans=1, seed=7)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(config.PARAMETRES, chemins_conditionnels=2), \
             patch.dict(config.PREPAIEMENT, intercept=-1, beta_change=0, beta_gain=0, almin=.01), \
             patch.object(config, 'SAUVER_SWAPS', True):
            tables = run.run_experiment(experiment)
            database = run.open_database(Path(tmp)/'batch.sqlite')
            run.save_experiment(database, experiment, tables)
            run.export_csv(database, Path(tmp))
            database.close()
            swaps = pd.read_csv(Path(tmp)/'swaps.csv')
            self.assertTrue((swaps.ferme_ce_mois==1).any())
            self.assertTrue(np.isfinite(swaps.prix_mtm_eur).all())
            monthly = pd.read_csv(Path(tmp)/'marges.csv')
            grouped = swaps.groupby(['strategie','date']).soulte_unwind_eur.sum()
            for _, row in monthly.iterrows():
                self.assertAlmostEqual(grouped.get((row.strategie,row.date),0.),row.soulte_unwind_eur,places=7)

    def test_conditional_fans_start_on_the_central(self):
        import tempfile
        import matplotlib.pyplot as plt
        from config import PARAMETRES
        from scripts.model import market
        from plot import plot_hw
        parameters = dict(PARAMETRES, seed=4, maturite_ans=1, chemins_conditionnels=2,
                          profil_initial='descendante', hw_a=.1, hw_sigma=.005)
        _, simulation = market(parameters)
        before = simulation.scenario_central.copy()
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)/'scenarios.png'
            figure = plot_hw(simulation, destination, nombre=8, decalages=(1,6))
            plt.close(figure)
            self.assertGreater(destination.stat().st_size,1000)
        np.testing.assert_array_equal(before.taux_6m,simulation.scenario_central.taux_6m)


def reference_swap_value(swap, paths):
    import pandas as pd
    from scripts.model import _schedule_arrays, _path_values_at, _forward_rates_vector
    valuation = paths.dates[0]
    count = len(paths.taux_6m)
    if (
        swap.trade_date > valuation
        or valuation >= swap.end_date
        or swap.closed_date is not None and swap.closed_date <= valuation
    ):
        return np.zeros(count)
    sign = 1 if swap.pay_fixed else -1
    starts, ends = _schedule_arrays(swap.start_date, swap.end_date, 1)
    keep = ends > valuation
    delta = (ends[keep] - starts[keep]).days.to_numpy() / 365
    factors = _path_values_at(paths, ends[keep], paths.facteurs_actualisation, True)
    fixed = -sign * swap.notional * swap.fixed_rate * np.sum(factors * delta[None, :], axis=1)
    starts, ends = _schedule_arrays(swap.start_date, swap.end_date, swap.floating_payment_months)
    keep = ends > valuation
    starts, ends = (starts[keep], ends[keep])
    rates = np.empty((count, len(ends)))
    known = np.array([d in swap.floating_fixings for d in starts])
    if known.any():
        rates[:, known] = np.array([swap.floating_fixings[d] for d in starts[known]])[None, :]
    if (~known).any():
        projected = starts[~known]
        fixing = projected if swap.fixing_lag_business_days == 0 else projected - pd.offsets.BDay(swap.fixing_lag_business_days)
        rates[:, ~known] = _forward_rates_vector(paths, fixing, projected, ends[~known])
    factors = _path_values_at(paths, ends, paths.facteurs_actualisation, True)
    delta = (ends - starts).days.to_numpy() / 360
    floating = sign * swap.notional * np.sum(rates * factors * delta[None, :], axis=1)
    return fixed + floating



class PricingTests(unittest.TestCase):
    def test_shared_swap_legs_match_direct_pricing(self):
        import pandas as pd
        from dataclasses import replace
        from config import PARAMETRES
        from scripts.model import Swap, market, pv_swap, update_known_fixings
        from scripts.hw import parallel_bump
        parameters = dict(PARAMETRES, seed=6, maturite_ans=1, chemins_conditionnels=3,
                          profil_initial='montante', hw_a=.1, hw_sigma=.01)
        _, simulation = market(parameters)
        trade = simulation.scenario_central.index[0]
        start = trade+pd.offsets.BDay(4)
        book = {
            str(i): Swap(str(i), start, pd.Timestamp(end), 100000.+i*5000, rate, pay, trade,
                         fixing_lag_business_days=2)
            for i, (end, rate, pay) in enumerate([
                ('2026-01-31', .03, True), ('2026-01-31', .05, False),
                ('2025-06-30', .02, True), ('2025-10-31', .04, False)])
        }
        for date in simulation.scenario_central.index[:-1]:
            update_known_fixings(book, date, simulation)
            paths = simulation.scenarios_conditionnels[date.strftime('%Y%m')]
            for current in [paths, parallel_bump(paths, 1e-4)]:
                for swap in book.values():
                    np.testing.assert_allclose(pv_swap(swap,current),reference_swap_value(swap,current),atol=1e-8)

    def test_parallel_bump_preserves_the_shape_of_each_curve(self):
        from config import PARAMETRES
        from scripts.hw import parallel_bump
        from scripts.model import market
        times = np.array([.1, .5, 1., 5., 20.])
        for profile in ['plate','montante','descendante']:
            parameters = dict(PARAMETRES, seed=1, maturite_ans=1, chemins_conditionnels=2,
                              profil_initial=profile, hw_a=.1, hw_sigma=.005)
            _, simulation = market(parameters)
            paths = list(simulation.scenarios_conditionnels.values())[5]
            bumped = parallel_bump(paths, 1e-4)
            np.testing.assert_allclose(bumped.courbe.zero(times)-paths.courbe.zero(times),1e-4,atol=1e-14)
            np.testing.assert_allclose(bumped.courbe.forward_instantane(times)-paths.courbe.forward_instantane(times),1e-4,atol=1e-14)
            np.testing.assert_allclose(bumped.courbe.discount(times)/paths.courbe.discount(times),np.exp(-1e-4*times),atol=1e-14)
