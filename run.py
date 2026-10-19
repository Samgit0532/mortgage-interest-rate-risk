"""Lancer le batch de prêts fixes et révisables : python run.py. Reprise automatique."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

import config
from scripts.model import PrepaymentSettings, STRATEGIES, market, strategy
from scripts.variable import variable_strategy

ROOT = Path(__file__).resolve().parent


def configuration_key():
    """Séparer aussi les résultats si les hypothèses communes changent."""
    content = {
        'parametres': config.PARAMETRES,
        'prepaiement': config.PREPAIEMENT,
        'courbes': config.PROFILS,
        'echelle_courbe': config.ECHELLE_COURBE,
        'sauver_swaps': config.SAUVER_SWAPS,
        'produits': config.PRODUITS,
        'variable': config.VARIABLE,
    }
    digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode())
    for path in [ROOT / 'scripts' / 'model.py', ROOT / 'scripts' / 'hw.py', ROOT / 'scripts' / 'variable.py', ROOT / 'run.py']:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def experience_key(experiment):
    content = {'jeu': configuration_key(), **experiment}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:24]


def run_experiment(experiment):
    parameters = {
        **config.PARAMETRES,
        'profil_initial': experiment['profil'],
        'hw_a': experiment['a'],
        'hw_sigma': experiment['sigma'],
        'maturite_ans': experiment['maturite_ans'],
        'seed': experiment['seed'],
    }
    context = market(parameters)
    settings = PrepaymentSettings(**config.PREPAIEMENT)
    margins, swap_books, option_books = [], [], []
    if 'fixe' in config.PRODUITS:
        for name, expected, unwind in STRATEGIES:
            monthly, swaps = strategy(
                parameters, context, name, expected, unwind,
                prepayment_settings=settings, save_swaps=config.SAUVER_SWAPS,
            )
            monthly['produit'] = 'fixe'
            margins.append(monthly)
            if config.SAUVER_SWAPS:
                swap_books.append(swaps)
    for product, protected in [('variable', False), ('protected', True)]:
        if product not in config.PRODUITS:
            continue
        monthly, swaps, options = variable_strategy(
            parameters, context, config.VARIABLE,
            protected=protected, save_swaps=config.SAUVER_SWAPS,
        )
        monthly['produit'] = product
        margins.append(monthly)
        if config.SAUVER_SWAPS:
            swap_books.append(swaps)
        if not options.empty:
            option_books.append(options)
    if not margins:
        raise ValueError('Choisir au moins un produit parmi fixe, variable, protected.')

    monthly = pd.concat(margins, ignore_index=True)
    for column in ['prime_swaptions_payee_eur', 'cout_protection_lisse_eur', 'supplement_client_protection_bp']:
        if column not in monthly:
            monthly[column] = 0.
        monthly[column] = monthly[column].fillna(0.)
    for column, original in [('marge_lissee_eur', 'marge_mois_eur'), ('marge_cumulee_lissee_eur', 'marge_cumulee_eur')]:
        if column not in monthly:
            monthly[column] = monthly[original]
        monthly[column] = monthly[column].fillna(monthly[original])
    monthly['taux_6m_central_pct'] = monthly.date.map(context[1].scenario_central.taux_6m) * 100
    summary = monthly.groupby('strategie', sort=False).agg(
        marge_cumulee_finale_eur=('marge_cumulee_eur', 'last'),
        marge_cumulee_lissee_finale_eur=('marge_cumulee_lissee_eur', 'last'),
        prime_swaptions_totale_eur=('prime_swaptions_payee_eur', 'sum'),
        pire_mois_eur=('marge_mois_eur', 'min'),
        marge_moyenne_mois_eur=('marge_mois_eur', 'mean'),
        nombre_prepaiements=('principal_prepaye_eur', lambda s: int((s > 0).sum())),
        notionnel_brut_max_eur=('notionnel_brut_eur', 'max'),
        bpv_residuelle_max_eur_par_pb=('bpv_nette_eur_par_pb', lambda s: s.abs().max()),
    ).reset_index()
    summary['marge_moyenne_mois_eur'] = summary.marge_cumulee_finale_eur / (12 * parameters['maturite_ans'])
    swaps = pd.concat(swap_books, ignore_index=True) if swap_books else pd.DataFrame()
    metadata = {'experience': experience_key(experiment), 'jeu': configuration_key(), **experiment}
    tables = {'marges': monthly, 'resume': summary, 'swaps': swaps, 'swaptions': pd.concat(option_books, ignore_index=True) if option_books else pd.DataFrame()}
    for name, table in tables.items():
        if table.empty:
            continue
        first = ['strategie', 'date', 'marge_mois_eur', 'marge_cumulee_eur']
        columns = [column for column in first if column in table]
        columns += [column for column in table if column not in columns]
        tables[name] = table.assign(**metadata)[list(metadata) + columns]
    return tables


def open_database(path):
    database = sqlite3.connect(path)
    database.execute('''CREATE TABLE IF NOT EXISTS experiences (
        experience TEXT PRIMARY KEY,
        jeu TEXT,
        profil TEXT, a REAL, sigma REAL, maturite_ans INTEGER, seed INTEGER,
        statut TEXT, message TEXT, configuration TEXT
    )''')
    database.commit()
    return database


def sql_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value.item() if isinstance(value, np.generic) else value


def insert_table(database, name, table):
    if table.empty:
        return
    columns = []
    for column, dtype in table.dtypes.items():
        kind = 'REAL' if pd.api.types.is_numeric_dtype(dtype) else 'TEXT'
        if pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            kind = 'INTEGER'
        columns.append(f'"{column}" {kind}')
    database.execute(f'CREATE TABLE IF NOT EXISTS "{name}" ({", ".join(columns)})')
    existing = {row[1] for row in database.execute(f'PRAGMA table_info("{name}")')}
    for definition, column in zip(columns, table.columns):
        if column not in existing:
            database.execute(f'ALTER TABLE "{name}" ADD COLUMN {definition}')
    placeholders = ', '.join('?' for _ in table.columns)
    names = ', '.join(f'"{column}"' for column in table.columns)
    records = (tuple(sql_value(v) for v in row) for row in table.itertuples(index=False, name=None))
    database.executemany(f'INSERT INTO "{name}" ({names}) VALUES ({placeholders})', records)


def record_experiment(database, experiment, status, message=''):
    metadata = (
        experience_key(experiment), configuration_key(), experiment['profil'], experiment['a'],
        experiment['sigma'], experiment['maturite_ans'], experiment['seed'], status,
        message, json.dumps({'parametres': config.PARAMETRES, 'prepaiement': config.PREPAIEMENT,
                             'courbes': config.PROFILS, 'echelle_courbe': config.ECHELLE_COURBE,
                             'sauver_swaps': config.SAUVER_SWAPS, 'produits': config.PRODUITS, 'variable': config.VARIABLE}, sort_keys=True),
    )
    database.execute('INSERT OR REPLACE INTO experiences VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', metadata)


def save_experiment(database, experiment, tables):
    # Données et statut sont validés ensemble : une interruption ne crée pas de doublon.
    with database:
        database.execute('BEGIN IMMEDIATE')
        previous = database.execute(
            'SELECT statut FROM experiences WHERE experience = ?',
            (experience_key(experiment),),
        ).fetchone()
        if previous and previous[0] == 'terminee':
            return
        for name, table in tables.items():
            insert_table(database, name, table)
        record_experiment(database, experiment, 'terminee')


def export_csv(database, output):
    tables = {row[0] for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for name in ['experiences', 'marges', 'resume', 'swaps', 'swaptions']:
        if name not in tables:
            continue
        temporary = output / f'{name}.csv.tmp'
        with temporary.open('w', newline='', encoding='utf-8') as stream:
            cursor = database.execute(f'SELECT * FROM "{name}"')
            writer = csv.writer(stream)
            writer.writerow([column[0] for column in cursor.description])
            while rows := cursor.fetchmany(10_000):
                writer.writerows(rows)
        temporary.replace(output / f'{name}.csv')


def run_batch(output, limit=None):
    if limit is not None and limit < 1:
        raise ValueError('--limit doit être positif.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    experiments = list(config.experiences())
    print(f'{len(experiments)} expériences configurées ; produits : {config.PRODUITS}.', flush=True)
    database = open_database(output / 'batch.sqlite')
    completed = {row[0] for row in database.execute("SELECT experience FROM experiences WHERE statut='terminee'")}
    unique = {experience_key(e): e for e in experiments}
    pending = [e for key, e in unique.items() if key not in completed]
    selected = pending if limit is None else pending[:limit]
    print(f'{len(pending)} à faire ; {len(selected)} lancées cette fois.', flush=True)
    failures = 0
    started = time.monotonic()
    try:
        for i, experiment in enumerate(selected, start=1):
            print(f"[{i}/{len(selected)}] {experiment}", flush=True)
            try:
                tables = run_experiment(experiment)
                save_experiment(database, experiment, tables)
            except Exception as error:
                failures += 1
                with database:
                    record_experiment(database, experiment, 'erreur', f'{type(error).__name__}: {error}')
                print(f'  Échec : {error}', file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        print('\nBatch interrompu. Les expériences terminées sont conservées.', flush=True)
        raise
    finally:
        try:
            export_csv(database, output)
        finally:
            database.close()
    print(f'Fin en {(time.monotonic() - started) / 60:.1f} min. Résultats : {output.resolve()}')
    return failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'resultats')
    parser.add_argument('--limit', type=int, help='Limiter le nombre de nouvelles expériences')
    parser.add_argument('--export', action='store_true', help='Reconstruire les CSV sans simuler')
    parser.add_argument('--test', action='store_true', help='Exécuter seulement les tests')
    args = parser.parse_args()
    if args.test:
        import unittest
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.export:
        if not (args.output / 'batch.sqlite').exists():
            parser.error('Aucun batch.sqlite dans ce dossier.')
        with open_database(args.output / 'batch.sqlite') as database:
            export_csv(database, args.output)
    else:
        try:
            raise SystemExit(1 if run_batch(args.output, args.limit) else 0)
        except ValueError as error:
            parser.exit(2, f'{error}\n')
        except KeyboardInterrupt:
            raise SystemExit(130)
