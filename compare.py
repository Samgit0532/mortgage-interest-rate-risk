"""Comparer les stratégies du batch, sans simulation ni modification du SQLite.

Placer ce fichier à côté de run.py, puis lancer :
    python compare.py
    python compare.py --input resultats/batch.sqlite --maturite 20
    python compare.py --mesure lissee

Dépendances : numpy et pandas. Les CSV s'ouvrent dans Excel (séparateur ;).
"""

import argparse
from html import escape
from itertools import combinations, groupby
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
GROUP = ['jeu', 'profil', 'a', 'sigma', 'maturite_ans']
LABELS = {
    'fin_contractuelle_unwind': 'Fin contractuelle / unwind',
    'fin_contractuelle_conservation': 'Fin contractuelle / conservation',
    'fin_esperee_unwind': 'Fin espérée / unwind',
    'fin_esperee_conservation': 'Fin espérée / conservation',
    'variable_normal': 'Variable 3/3/3',
    'variable_protected': 'Variable protected',
}
TOLERANCE_EUR = 1e-6


def load_metrics(path, maturity=None, game=None, measure='cash'):
    """Lire un seul échéancier à la fois : mémoire indépendante du nombre de mois du batch."""
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f'Base SQLite introuvable : {path}')
    if measure not in {'cash', 'lissee'}:
        raise ValueError('La mesure doit être cash ou lissee.')
    amount, cumulative = ('marge_mois_eur', 'marge_cumulee_eur')
    if measure == 'lissee':
        amount, cumulative = ('marge_lissee_eur', 'marge_cumulee_lissee_eur')

    records = []
    # Le mode lecture seule interdit notamment de créer une base par erreur.
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
        db.execute('PRAGMA query_only = ON')
        db.execute('BEGIN')
        columns = {row[1] for row in db.execute('PRAGMA table_info(marges)')}
        wanted = ['experience', 'strategie', *GROUP, 'seed', 'date', amount,
                  cumulative, 'encours_debut_eur']
        missing = set(wanted) - columns
        if missing:
            raise ValueError(f'Colonnes absentes de marges : {", ".join(sorted(missing))}')

        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        manifest = {}
        if 'experiences' in tables:
            manifest = {
                key: (status, configuration)
                for key, status, configuration in db.execute(
                    'SELECT experience, statut, configuration FROM experiences'
                )
            }
        conditions, parameters = [], []
        if maturity is not None:
            conditions.append('maturite_ans = ?')
            parameters.append(maturity)
        if game is not None:
            conditions.append('jeu = ?')
            parameters.append(game)
        if 'experiences' in tables:
            conditions.append("experience IN (SELECT experience FROM experiences WHERE statut='terminee')")
        where = ' WHERE ' + ' AND '.join(conditions) if conditions else ''
        names = ', '.join(f'"{name}"' for name in wanted)
        cursor = db.execute(f'SELECT {names} FROM marges{where} ORDER BY experience, strategie, date', parameters)

        for (experiment, strategy), iterator in groupby(cursor, key=lambda row: row[:2]):
            frame = pd.DataFrame.from_records(list(iterator), columns=wanted)
            context = f'{experiment} / {strategy}'
            if frame[GROUP + ['seed']].isna().any().any():
                raise ValueError(f'Métadonnées manquantes : {context}')
            if len(frame[GROUP + ['seed']].drop_duplicates()) != 1:
                raise ValueError(f'Métadonnées contradictoires : {context}')
            values = frame[[amount, cumulative, 'encours_debut_eur']].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f'Marges ou encours manquants/non finis : {context}')
            dates = pd.DatetimeIndex(pd.to_datetime(frame.date, errors='raise'))
            if dates.has_duplicates or not dates.is_monotonic_increasing:
                raise ValueError(f'Dates dupliquées ou désordonnées : {context}')

            years = float(frame.maturite_ans.iloc[0])
            months = int(round(12 * years))
            if months < 1 or not np.isclose(months, 12 * years):
                raise ValueError(f'Maturité non exprimable en mois entiers : {context}')
            # Les prêts variables ont une ligne à l'origine (achat des options).
            # Elle ne doit pas ajouter un faux mois dans la moyenne ou le pire mois.
            configuration = manifest.get(experiment, ('', None))[1]
            origin_month = None
            if configuration:
                config = json.loads(configuration)
                start = config.get('parametres', {}).get('debut')
                if start is not None:
                    origin_month = pd.to_datetime(str(start), format='%Y%m').to_period('M')
            if origin_month is None:
                origin_month = dates[-1].to_period('M') - months
            periods = dates.to_period('M')
            initial = periods == origin_month
            expected = pd.period_range(origin_month + 1, periods=months, freq='M')
            if initial.sum() > 1 or not periods[~initial].equals(expected):
                raise ValueError(f'Échéancier incomplet ou mois dupliqués : {context}')
            if initial.any() and abs(values[initial, 2][0]) > TOLERANCE_EUR:
                raise ValueError(f'La ligne initiale contient un encours début inattendu : {context}')
            cash, total = values[:, 0], values[:, 1]
            if not np.allclose(np.cumsum(cash), total, atol=0.01, rtol=1e-9):
                raise ValueError(f'Le cumul ne correspond pas à la somme des marges : {context}')
            monthly = cash[~initial]
            monthly_dates = dates[~initial]
            running_peak = np.maximum.accumulate(np.r_[0., total])[1:]
            worst = int(np.argmin(monthly))
            record = frame.iloc[0][['experience', *GROUP, 'seed', 'strategie']].to_dict()
            record.update(
                nombre_mois=months,
                marge_finale_eur=float(total[-1]),
                marge_moyenne_mois_eur=float(total[-1] / months),
                marge_moyenne_hors_origine_eur=float(monthly.mean()),
                flux_origine_eur=float(cash[initial].sum()),
                pire_mois_eur=float(monthly[worst]),
                date_pire_mois=monthly_dates[worst].strftime('%Y-%m-%d'),
                perte_max_un_mois_eur=float(max(0., -monthly[worst])),
                pire_flux_origine_incluse_eur=float(cash.min()),
                proportion_mois_negatifs=float(np.mean(monthly < 0)),
                baisse_max_cumul_eur=float(np.max(running_peak - total)),
            )
            records.append(record)
    if not records:
        raise ValueError('Aucune expérience terminée ne correspond aux filtres.')
    result = pd.DataFrame(records)
    if result.duplicated(GROUP + ['seed', 'strategie']).any():
        raise ValueError('Plusieurs expériences pour les mêmes paramètres, seed et stratégie.')
    return result


def compare(metrics):
    """Comparer chaque stratégie sur exactement les mêmes expériences au sein d'un groupe."""
    summaries, pairs, exclusions = [], [], []
    for key, group in metrics.groupby(GROUP, sort=True):
        metadata = dict(zip(GROUP, key))
        strategies = sorted(group.strategie.unique())
        # Une stratégie absente d'une expérience ne doit pas favoriser les autres.
        counts = group.groupby('experience').strategie.nunique()
        common = counts.index[counts == len(strategies)]
        excluded = group.loc[~group.experience.isin(common)].copy()
        if not excluded.empty:
            excluded['raison'] = 'Expérience incomplète : certaines stratégies du groupe sont absentes.'
            exclusions.append(excluded)
        matched = group[group.experience.isin(common)]
        if matched.empty:
            continue
        matrix = matched.pivot(index='experience', columns='strategie', values='marge_finale_eur')
        winners = matrix.sub(matrix.max(axis=1), axis=0).abs() <= TOLERANCE_EUR
        # Une égalité entre quatre stratégies donne 1/4 de victoire à chacune.
        win_share = winners.div(winners.sum(axis=1), axis=0).mean()

        for name, rows in matched.groupby('strategie'):
            final = rows.marge_finale_eur
            summaries.append({
                **metadata,
                'strategie': name,
                'nombre_seeds': len(rows),
                'experiences_exclues': len(counts) - len(common),
                'marge_finale_moyenne_eur': final.mean(),
                'marge_finale_mediane_eur': final.median(),
                'dispersion_marge_finale_eur': final.std(ddof=1),
                'marge_finale_q05_eur': final.quantile(.05),
                'marge_finale_q95_eur': final.quantile(.95),
                'pire_marge_finale_eur': final.min(),
                'frequence_finale_negative': (final < 0).mean(),
                'marge_moyenne_mois_eur': rows.marge_moyenne_mois_eur.mean(),
                'pire_mois_moyen_eur': rows.pire_mois_eur.mean(),
                'pire_mois_observe_eur': rows.pire_mois_eur.min(),
                'perte_max_un_mois_eur': rows.perte_max_un_mois_eur.max(),
                'proportion_mois_negatifs': rows.proportion_mois_negatifs.mean(),
                'baisse_max_cumul_moyenne_eur': rows.baisse_max_cumul_eur.mean(),
                'baisse_max_cumul_observee_eur': rows.baisse_max_cumul_eur.max(),
                'flux_origine_moyen_eur': rows.flux_origine_eur.mean(),
                'part_victoires': win_share[name],
            })

        for first, second in combinations(strategies, 2):
            difference = matrix[first] - matrix[second]
            pairs.append({
                **metadata, 'strategie_A': first, 'strategie_B': second,
                'nombre_seeds': len(difference),
                'ecart_moyen_A_moins_B_eur': difference.mean(),
                'ecart_median_A_moins_B_eur': difference.median(),
                'dispersion_ecart_eur': difference.std(ddof=1),
                'ecart_q05_eur': difference.quantile(.05),
                'ecart_q95_eur': difference.quantile(.95),
                'frequence_A_superieure_B': (difference > TOLERANCE_EUR).mean(),
                'frequence_egalite': (difference.abs() <= TOLERANCE_EUR).mean(),
            })

    if not summaries:
        raise ValueError('Aucun groupe ne contient une expérience complète commune aux stratégies.')
    summary = pd.DataFrame(summaries)
    # Deux classements distincts : gain moyen et résultat dans le bas de la distribution.
    for column, rank in [('marge_finale_moyenne_eur', 'rang_gain_moyen'),
                         ('marge_finale_q05_eur', 'rang_prudence_q05')]:
        summary[rank] = summary.groupby(GROUP)[column].rank(method='min', ascending=False).astype(int)
    summary = summary.sort_values(GROUP + ['rang_gain_moyen', 'strategie'])
    excluded_columns = list(metrics.columns) + ['raison']
    excluded = pd.concat(exclusions, ignore_index=True) if exclusions else pd.DataFrame(columns=excluded_columns)
    pair_columns = GROUP + ['strategie_A', 'strategie_B', 'nombre_seeds',
                           'ecart_moyen_A_moins_B_eur', 'ecart_median_A_moins_B_eur',
                           'dispersion_ecart_eur', 'ecart_q05_eur', 'ecart_q95_eur',
                           'frequence_A_superieure_B', 'frequence_egalite']
    return summary, pd.DataFrame(pairs, columns=pair_columns), excluded


def write_report(summary, pairs, excluded, output, source, measure):
    """Un rapport lisible, avec un tableau par combinaison de paramètres."""
    def table(frame):
        frame = frame.copy().replace(LABELS)
        return frame.to_html(index=False, border=0, na_rep='—', float_format=lambda x: f'{x:,.2f}')

    parts = [f'''<!doctype html><html lang="fr"><meta charset="utf-8">
<title>Comparaison des stratégies</title>
<style>body{{font:15px system-ui;margin:32px;color:#172536;max-width:1500px}}
h1,h2{{color:#123c60}}table{{border-collapse:collapse;font-size:13px;white-space:nowrap}}
td,th{{padding:9px 12px;border-bottom:1px solid #ddd;text-align:right}}
th{{background:#edf3f8}}td:first-child,th:first-child{{text-align:left}}
.scroll{{overflow:auto;margin-bottom:24px}}p,li{{line-height:1.6}}small{{color:#536376}}</style>
<h1>Comparaison des stratégies</h1>
<p>Base : <code>{escape(str(source))}</code> · Mesure : <b>{escape(measure)}</b>.</p>
<p>Les stratégies sont comparées sur les mêmes seeds, séparément pour chaque jeu
d'hypothèses, profil de courbe, vitesse de retour a, volatilité sigma et maturité.
Les taux a et sigma des CSV conservent leurs unités du modèle.</p>
<ul>
<li><b>Gain moyen</b> : moyenne des marges cumulées finales, en euros non actualisés.</li>
<li><b>Dispersion</b> : écart type entre seeds. Indéfini avec une seule seed.</li>
<li><b>Q05</b> : quantile à 5 % des résultats finaux ; plus il est élevé, mieux c'est.
Il décrit les simulations, sans garantie sur les pertes futures.</li>
<li><b>Pire mois</b> : minimum des marges aux échéances mensuelles.
La dépense initiale des options est affichée séparément.</li>
<li><b>Baisse du cumul</b> : plus forte baisse depuis un sommet du cumul, en euros positifs ;
le point de départ avant toute dépense est zéro.</li>
<li><b>Moyenne par mois</b> : cumul final / nombre de mois du prêt, dépense initiale comprise.
À maturité identique, son classement est identique à celui du cumul final.</li>
<li><b>Victoires</b> : part des scénarios où la stratégie termine première ;
les égalités partagent une victoire. Ces parts totalisent 100 % dans un groupe.</li>
</ul>
<p>En mode cash, la prime des swaptions est décaissée à l'origine.
En mode lissé, on utilise les colonnes lissées enregistrées par le simulateur.
Les deux conventions ne sont jamais mélangées.</p>
<p>{len(excluded.experience.unique()) if not excluded.empty else 0} expérience(s)
incomplète(s) exclue(s) des comparaisons. Voir <code>exclusions.csv</code>.
Tous les résultats individuels restent dans <code>par_seed.csv</code>.</p>''']

    rename = {
        'strategie': 'Stratégie', 'nombre_seeds': 'Seeds',
        'rang_gain_moyen': 'Rang gain', 'rang_prudence_q05': 'Rang Q05',
        'marge_finale_moyenne_eur': 'Gain moyen €',
        'marge_finale_q05_eur': 'Q05 €', 'marge_finale_q95_eur': 'Q95 €',
        'dispersion_marge_finale_eur': 'Dispersion €',
        'marge_moyenne_mois_eur': 'Moyenne / mois €',
        'pire_mois_observe_eur': 'Pire mois observé €',
        'baisse_max_cumul_observee_eur': 'Baisse max cumul €',
        'flux_origine_moyen_eur': 'Flux initial moyen €',
        'frequence_finale_negative': 'Fin négative %', 'part_victoires': 'Victoires %',
    }
    for key, group in summary.groupby(GROUP, sort=True):
        game, profile, a, sigma, years = key
        parts.append(f'<h2>{escape(str(profile))} · a = {a:g} · σ = {sigma * 100:g} % · {years:g} ans</h2>')
        parts.append(f'<small>Jeu : {escape(str(game))}</small>')
        view = group[list(rename)].copy()
        for column in ['frequence_finale_negative', 'part_victoires']:
            view[column] *= 100
        parts.append('<div class="scroll">' + table(view.rename(columns=rename)) + '</div>')
        if not pairs.empty:
            selection = np.ones(len(pairs), dtype=bool)
            for column, value in zip(GROUP, key):
                selection &= pairs[column].eq(value).to_numpy()
            view = pairs.loc[selection, ['strategie_A', 'strategie_B',
                                        'ecart_moyen_A_moins_B_eur', 'ecart_q05_eur',
                                        'ecart_q95_eur', 'frequence_A_superieure_B']].copy()
            view['frequence_A_superieure_B'] *= 100
            view.columns = ['Stratégie A', 'Stratégie B', 'Écart moyen A − B €',
                            'Q05 écart €', 'Q95 écart €', 'A supérieure à B %']
            parts.append('<p>Écarts de cumul final sur les mêmes scénarios : un écart positif favorise A.</p>')
            parts.append('<div class="scroll">' + table(view) + '</div>')
    parts.append('</html>')
    (output / 'rapport.html').write_text('\n'.join(parts), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, default=ROOT / 'resultats' / 'batch.sqlite',
                        help='Base SQLite ou dossier contenant batch.sqlite')
    parser.add_argument('--output', type=Path, help='Dossier de sortie ; défaut : comparaison à côté de la base')
    parser.add_argument('--maturite', type=int, help='Filtrer une maturité en années')
    parser.add_argument('--jeu', help="Filtrer un jeu d'hypothèses")
    parser.add_argument('--mesure', choices=['cash', 'lissee'], default='cash')
    args = parser.parse_args()
    source = args.input / 'batch.sqlite' if args.input.is_dir() else args.input
    source = source.resolve()
    output = (args.output or source.parent / f'comparaison_{args.mesure}').resolve()
    try:
        metrics = load_metrics(source, args.maturite, args.jeu, args.mesure)
        summary, pairs, excluded = compare(metrics)
        output.mkdir(parents=True, exist_ok=True)
        for name, frame in [('par_seed', metrics), ('comparaison', summary),
                            ('ecarts_strategies', pairs), ('exclusions', excluded)]:
            frame.assign(mesure=args.mesure).to_csv(
                output / f'{name}.csv', index=False, sep=';', decimal=',', encoding='utf-8-sig'
            )
        write_report(summary, pairs, excluded, output, source, args.mesure)
    except (ValueError, sqlite3.Error, OSError) as error:
        parser.exit(1, f'Erreur : {error}\n')
    print(f'Comparaison enregistrée : {output / "rapport.html"}')


if __name__ == '__main__':
    main()
