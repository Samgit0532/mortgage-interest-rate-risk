"""Comparer les marges enregistrées : python plot.py --input resultats."""
import argparse
from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from scripts.hw import generer_chemins

LABELS = {
    'fin_contractuelle_unwind': 'Fin contractuelle / unwind',
    'fin_contractuelle_conservation': 'Fin contractuelle / conservation',
    'fin_esperee_unwind': 'Fin espérée / unwind',
    'fin_esperee_conservation': 'Fin espérée / conservation',
    'variable_normal': 'Variable 3/3/3',
    'variable_protected': 'Variable protected',
}
COLORS = ['#2563eb', '#08916f', '#e78124', '#9b51a0', '#cf304b', '#172033']
GROUPS = ['jeu', 'profil', 'a', 'sigma', 'maturite_ans']
PLOT_COLUMNS = GROUPS + ['experience', 'seed', 'strategie', 'date',
                        'marge_mois_eur', 'marge_cumulee_eur']


def read_results(folder):
    """La base permet aussi de tracer un batch encore en cours."""
    path = Path(folder) / 'batch.sqlite'
    if path.exists():
        database = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
        try:
            return pd.read_sql_query(f'SELECT {", ".join(PLOT_COLUMNS)} FROM marges', database)
        finally:
            database.close()
    return pd.read_csv(Path(folder) / 'marges.csv', usecols=PLOT_COLUMNS)


def result_groups(folder):
    """Lire un groupe à la fois pour ne pas charger tout le batch en mémoire."""
    path = Path(folder) / 'batch.sqlite'
    if not path.exists():
        yield from check_results(read_results(folder)).groupby(GROUPS, sort=True)
        return
    database = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        groups = database.execute(f'SELECT DISTINCT {", ".join(GROUPS)} FROM marges').fetchall()
        if not groups:
            raise ValueError('Aucune expérience terminée à tracer.')
        condition = ' AND '.join(f'{column} = ?' for column in GROUPS)
        query = f'SELECT {", ".join(PLOT_COLUMNS)} FROM marges WHERE {condition}'
        for group in groups:
            yield group, check_results(pd.read_sql_query(query, database, params=group))
    finally:
        database.close()


def check_results(table):
    required = set(GROUPS + ['experience', 'seed', 'strategie', 'date',
                            'marge_mois_eur', 'marge_cumulee_eur'])
    if not required <= set(table):
        raise ValueError('Le fichier ne contient pas toutes les colonnes de marges.')
    if table.empty:
        raise ValueError('Aucune expérience terminée à tracer.')
    if table.duplicated(['experience', 'strategie', 'date']).any():
        raise ValueError('Doublons détectés dans les résultats.')
    table = table.copy()
    table['date'] = pd.to_datetime(table['date'])
    return table


def margin_figure(data, column, title, path):
    if data.seed.nunique() != 1:
        raise ValueError('Ce graphique doit contenir une seule seed.')
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for (name, label), color in zip(LABELS.items(), COLORS):
        rows = data.loc[data.strategie == name].sort_values('date')
        if rows.empty:
            continue
        ax.plot(rows.date, rows[column], color=color, label=label, lw=1.6,
                ls='--' if 'conservation' in name else '-')
    ax.axhline(0, color='#475569', lw=.7)
    ax.set(title=title, xlabel='Date', ylabel='Euros')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def paired_differences(data, reference):
    """Soustraire les stratégies à expérience et mois identiques, avant agrégation."""
    base = data.loc[data.strategie == reference, ['experience', 'date', 'marge_cumulee_eur']]
    base = base.rename(columns={'marge_cumulee_eur': 'reference_eur'})
    joined = data.merge(base, on=['experience', 'date'], how='left', validate='many_to_one')
    if joined.reference_eur.isna().any():
        raise ValueError('La stratégie de référence manque pour certaines expériences.')
    joined['ecart_cumule_eur'] = joined.marge_cumulee_eur - joined.reference_eur
    return joined


def read_group_table(folder, name, group, seed):
    database_path = Path(folder) / 'batch.sqlite'
    if database_path.exists():
        database = sqlite3.connect(database_path.resolve().as_uri() + '?mode=ro', uri=True)
        try:
            exists = database.execute("SELECT name FROM sqlite_master WHERE name=?", (name,)).fetchone()
            if exists is None:
                return pd.DataFrame()
            condition = ' AND '.join(f'"{column}"=?' for column in GROUPS)
            return pd.read_sql_query(f'SELECT * FROM "{name}" WHERE {condition} AND seed=?',
                                     database, params=(*group, seed))
        finally:
            database.close()
    path = Path(folder) / f'{name}.csv'
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path)
    for column, value in zip(GROUPS, group):
        table = table.loc[table[column] == value]
    return table.loc[table.seed == seed].copy()


def plot_batch(folder, output=None, reference=None, *, seed):
    """Une seule seed : marges, central réalisé et tableaux des transactions."""
    if type(seed) is not int or seed < 0:
        raise ValueError('Choisir une seed entière positive ou nulle.')
    output = Path(output) if output is not None else Path(folder) / f'figures_seed_{seed}'
    output.mkdir(parents=True, exist_ok=True)
    exported = 0

    def save(table, path):
        table.to_csv(path, index=False, sep=';', decimal=',', encoding='utf-8-sig',
                     date_format='%d/%m/%Y')

    for group, existing in result_groups(folder):
        if seed not in existing.seed.to_numpy():
            continue
        jeu, profil, a, sigma, maturity = group
        data = check_results(read_group_table(folder, 'marges', group, seed))
        destination = output / f'{jeu}_{profil}_a{a:g}_sigma{sigma:g}_{maturity}ans'
        destination.mkdir(parents=True, exist_ok=True)
        title = f'{profil} · a={a:g}, σ={sigma:g} · {maturity} ans · seed {seed}'
        for column, name, label in [
            ('marge_mois_eur', 'mensuelle', 'Marge cash mensuelle'),
            ('marge_cumulee_eur', 'cumulee', 'Marge cash cumulée'),
        ]:
            margin_figure(data, column, f'{label} · {title}', destination/f'marge_{name}.png')
            wide = data.pivot(index='date', columns='strategie', values=column).sort_index()
            wide = wide.rename(columns=LABELS)
            # Les stratégies fixes ne stockaient pas la ligne d'origine (cash nul).
            wide.loc[wide.index.min()] = wide.loc[wide.index.min()].fillna(0.)
            save(wide.reset_index(), destination/f'marges_{name}.csv')
        if 'prime_swaptions_payee_eur' in data and data.prime_swaptions_payee_eur.sum() > 0:
            for column, name in [('marge_lissee_eur', 'mensuelle'), ('marge_cumulee_lissee_eur', 'cumulee')]:
                margin_figure(data, column, f'Marge après coût de protection lissé · {title}',
                              destination/f'marge_{name}_lissee.png')
        save(data.sort_values(['strategie', 'date']), destination/'detail_marges.csv')

        if 'taux_6m_central_pct' in data and data.taux_6m_central_pct.notna().any():
            central = data[['date','taux_6m_central_pct']].dropna().drop_duplicates('date').sort_values('date')
            central = central.rename(columns={'taux_6m_central_pct':'taux_6m_pct'})
        else:
            central = data[['date','taux_financement_pct']].drop_duplicates('date').copy()
            central['date'] -= pd.offsets.MonthEnd(1)
            central = central.rename(columns={'taux_financement_pct':'taux_6m_pct'})
        save(central, destination/'scenario_central.csv')
        fig, ax = plt.subplots(figsize=(11,5))
        ax.plot(central.date, central.taux_6m_pct, color='#172033', lw=2)
        ax.set(title=f'Scénario central réalisé · {title}', xlabel='Date', ylabel='Taux 6M (%)')
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(destination/'scenario_central.png', dpi=160)
        plt.close(fig)

        swaps = read_group_table(folder, 'swaps', group, seed)
        if not swaps.empty:
            swaps['date'] = pd.to_datetime(swaps.date)
            swaps['date_transaction'] = pd.to_datetime(swaps.date_transaction)
            trades = swaps.loc[swaps.date == swaps.date_transaction].drop_duplicates(['strategie','swap_id'])
            columns = ['strategie','date_transaction','swap_id','origine','convention','sens','notionnel_eur',
                       'taux_fixe_pct','debut','echeance','prix_mtm_eur','bpv_eur_par_pb']
            save(trades[[c for c in columns if c in trades]].sort_values(['strategie','date_transaction']),
                 destination/'swaps_pris.csv')
            save(swaps.sort_values(['strategie','swap_id','date']), destination/'suivi_swaps.csv')
        else:
            print('Swaps non enregistrés pour ce groupe : activer SAUVER_SWAPS puis relancer le calcul.')
        options = read_group_table(folder, 'swaptions', group, seed)
        if not options.empty:
            ordered = options.sort_values(['option_id','date'])
            save(ordered, destination/'suivi_swaptions.csv')
            save(ordered.drop_duplicates('option_id'), destination/'swaptions_achetees.csv')
        print(f'Résultats : {destination.resolve()}')
        exported += 1
    if not exported:
        raise ValueError(f'Aucun résultat pour la seed {seed}.')


def plot_hw(simulation, output=None, *, nombre=200, decalages=(1, 120), seed=20250131):
    """Reprojette deux états conditionnels jusqu'à la fin du scénario central.

    La courbe et les paramètres sont celles connues à chaque date de départ.
    Une seed de dessin séparée laisse le scénario central inchangé.
    Retourne la figure ; output peut être un chemin PNG, SVG ou PDF.
    """
    if not isinstance(nombre,int) or nombre<1:
        raise ValueError('nombre doit être un entier positif.')
    if len(decalages)!=2 or any(not isinstance(m,int) or m<1 for m in decalages):
        raise ValueError('Fournir deux décalages positifs en mois.')
    if len(set(decalages))!=2:
        raise ValueError('Les deux dates de départ doivent être différentes.')
    central=simulation.scenario_central.sort_index()
    dates=pd.DatetimeIndex(central.index)
    if len(dates)<2 or not dates.is_unique or not np.all(np.isfinite(central.taux_6m)):
        raise ValueError('Scénario central invalide.')
    start,end=dates[0],dates[-1]
    colors=['#2563eb','#d97706']
    rngs=np.random.SeedSequence(seed).spawn(2)
    fans=[]
    for offset,color,rng in zip(decalages,colors,rngs):
        origin=start+pd.offsets.MonthEnd(offset)
        if origin>=end or origin not in dates:
            raise ValueError(f'Départ {origin:%Y-%m-%d} absent ou à la fin de la simulation.')
        known=simulation.scenarios_conditionnels.get(origin.strftime('%Y%m'))
        if known is None or known.courbe is None or known.parameters is None:
            raise ValueError(f'Courbe/parameters conditionnelle manquante au {origin:%Y-%m-%d}.')
        projected=generer_chemins(known.courbe,end,known.parameters,nombre,np.random.default_rng(rng))
        # Dessiner aux mêmes dates mensuelles que le central, sans les points de fixing.
        future_dates=dates[dates>=origin]
        indices=pd.DatetimeIndex(projected.dates).get_indexer(future_dates)
        if np.any(indices<0):raise ValueError('Dates mensuelles absentes des projections.')
        rates=projected.taux_6m[:,indices]*100
        anchor=float(central.loc[origin,'taux_6m'])*100
        if not np.allclose(rates[:,0],anchor,atol=1e-8,rtol=0):
            raise ValueError('Le faisceau ne part pas du taux central : vérifier la courbe conditionnelle.')
        fans.append((origin,future_dates,rates,color,offset))

    with plt.rc_context({'font.family':'DejaVu Sans','font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False}):
        fig,ax=plt.subplots(figsize=(13,6.8),facecolor='white')
        ax.set_facecolor('#fbfcfe')
        legend=[Line2D([0],[0],color='#172033',lw=2.4,label='Scénario central simulé')]
        for origin,future_dates,rates,color,offset in fans:
            x=mdates.date2num(future_dates.to_pydatetime())
            count=len(x)-1
            # Transparence décroissante avec l'horizon, ajustée au nombre de chemins.
            alpha_start=min(.23,1.25/np.sqrt(nombre))
            alpha=np.linspace(alpha_start,alpha_start*.12,count)
            rgba=np.tile(to_rgba(color),(count,1));rgba[:,3]=alpha
            segments=[]
            for path in rates:
                points=np.column_stack([x,path])
                segments.append(np.stack([points[:-1],points[1:]],axis=1))
            cloud=LineCollection(np.concatenate(segments),colors=np.tile(rgba,(nombre,1)),
                linewidths=.65,rasterized=True,zorder=2)
            ax.add_collection(cloud)
            # Repère central du faisceau ; la moyenne n'est pas une prévision garantie.
            ax.plot(future_dates,rates.mean(axis=0),color=color,ls=(0,(4,3)),lw=1.2,alpha=.7,zorder=3)
            ax.axvline(origin,color=color,lw=1,alpha=.2,ls=':',zorder=1)
            ax.scatter([origin],[rates[0,0]],s=65,color=color,edgecolor='white',lw=1.6,zorder=6)
            legend.append(Line2D([0],[0],color=color,lw=2,marker='o',
                label=f'{nombre} futurs à M+{offset} · {origin:%m/%Y}'))
        ax.plot(dates,central.taux_6m*100,color='#172033',lw=2.4,zorder=5)
        legend.append(Line2D([0],[0],color='#64748b',ls='--',lw=1.2,label='Moyenne des futurs (pointillés)'))
        ax.autoscale_view();ax.set_xlim(start,end)
        ax.margins(y=.12)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100,decimals=2))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5,maxticks=9))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.set_xlabel('Date',labelpad=10);ax.set_ylabel('Taux de référence 6M',labelpad=10)
        ax.grid(axis='y',color='#dce2ea',alpha=.65,lw=.7)
        ax.grid(axis='x',color='#e8edf3',alpha=.5,lw=.6)
        ax.tick_params(colors='#536176')
        ax.legend(handles=legend,loc='upper left',frameon=True,facecolor='white',
                  edgecolor='#e5e9f0',framealpha=.96,fontsize=9)
        fig.suptitle('Hull–White : futurs possibles à deux instants',x=.085,ha='left',
                     fontsize=19,fontweight='bold',color='#172033',y=.98)
        fig.text(.085,.925,'Deux couleurs · départ sur le central · effacement progressif jusqu’à l’échéance',
                 fontsize=10,color='#64748b')
        fig.text(.085,.025,'Chaque faisceau utilise la courbe et la paramètres disponibles à sa date de départ.',
                 fontsize=9,color='#64748b')
        fig.subplots_adjust(left=.095,right=.985,bottom=.13,top=.86)
        if output is not None:
            destination=Path(output);destination.parent.mkdir(parents=True,exist_ok=True)
            fig.savefig(destination,dpi=180,facecolor=fig.get_facecolor())
    return fig

def scenario_example(output, profil='plate', seed=1, maturity=15, a=.1, sigma=.005):
    """Illustration avec la configuration actuelle, sans calculer de marges."""
    import config
    from scripts.model import market
    parameters = dict(config.PARAMETRES, profil_initial=profil, seed=seed,
                      maturite_ans=maturity, hw_a=a, hw_sigma=sigma,
                      chemins_conditionnels=1)
    _, simulation = market(parameters)
    offsets = (1, 120) if maturity > 10 else (1, max(2, maturity*6))
    path = Path(output) / f'scenario_{profil}_seed{seed}.png'
    figure = plot_hw(simulation, path, decalages=offsets, seed=seed+20250131)
    plt.close(figure)
    print(path.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(__file__).resolve().parent / 'resultats')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--scenarios', action='store_true', help='Tracer seulement un central et ses faisceaux HW')
    parser.add_argument('--profil', choices=['plate', 'montante', 'descendante'], default='plate')
    parser.add_argument('--seed', type=int, required=True, help='Seed unique à afficher')
    parser.add_argument('--maturite', type=int, default=15)
    parser.add_argument('--a', type=float, default=.1)
    parser.add_argument('--sigma', type=float, default=.005)
    parser.add_argument('--reference', choices=LABELS, default='fin_contractuelle_conservation')
    args = parser.parse_args()
    try:
        if args.scenarios:
            scenario_example(args.output or args.input/'figures', args.profil, args.seed, args.maturite, args.a, args.sigma)
        else:
            plot_batch(args.input, args.output, args.reference, seed=args.seed)
    except (ValueError, OSError, sqlite3.Error, pd.errors.DatabaseError) as error:
        parser.exit(2, f'{error}\n')
