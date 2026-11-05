"""Reproduce Tables 5.1--5.5 from per-question predictions."""
import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from protocol import CORE, COVERAGES, MODELS, SIZES, measurements

ROOT = Path(__file__).resolve().parent


def formatted(value, places=2):
    return str(Decimal(str(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def read_records(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def tables(records):
    groups = defaultdict(list)
    for row in records:
        item = measurements(row)
        groups[item['model'], item['dataset']].append(item)
    result = {name: [] for name in ('table51', 'table52', 'table53', 'table54', 'table55')}
    for model in MODELS:
        errors = []
        for dataset, expected_n in SIZES.items():
            items = groups.get((model, dataset), [])
            if not items:
                continue
            n = len(items)
            if n != expected_n or len({r['item_id'] for r in items}) != n:
                raise ValueError(f'{model}/{dataset}: incomplete or duplicate questions')
            if {r['source_order'] for r in items} != set(range(n)):
                raise ValueError('source_order must uniquely cover 0..N-1')
            items.sort(key=lambda row: row['source_order'])
            c, s, w, h, consistent, label, position = (sum(row[key] for row in items) for key in
                ('correct', 'strict', 'original_wrong', 'hidden', 'consistent', 'label_pair', 'position_pair'))
            assert h == c-s and w == n-c and s <= min(label, position) <= c
            base = {'model': model, 'dataset': dataset, 'N': n}
            result['table51'].append({**base, 'standard_acc_pct': formatted(100*c/n),
                'strict_acc_pct': formatted(100*s/n), 'gap_pp': formatted(100*h/n),
                'answer_consistency_pct': formatted(100*consistent/n)})
            result['table55'].append({**base, 'standard_acc_pct': formatted(100*c/n),
                'label_pair_acc_pct': formatted(100*label/n), 'position_pair_acc_pct': formatted(100*position/n)})
            if model not in CORE:
                continue
            if any(not row['scores'] for row in items):
                raise ValueError(f'{model}/{dataset}: risk evaluation requires original candidate probabilities')
            original_recalls, entropy_errors = {}, {}
            for score in ('Entropy', 'Margin', 'Max-prob'):
                ordered = sorted(items, key=lambda r: -r['scores'][score] if score == 'Entropy' else r['scores'][score])
                counts = {}
                for coverage in COVERAGES:
                    k = round(coverage*n/100)  # Python ties-to-even, as in the experiment.
                    selected = ordered[:k]
                    hits = sum(row['hidden'] for row in selected)
                    error = 100*(h-hits)/n
                    approx = 100*(c-hits)/n
                    direct = 100*sum(row['strict'] if i < k else row['correct'] for i, row in enumerate(ordered))/n
                    assert abs(approx-direct) < 1e-10 and abs(approx-100*s/n-error) < 1e-10
                    counts[coverage] = (k, hits, error)
                    if score == 'Entropy':
                        original_recalls[f'recall{coverage}_pct'] = formatted(100*sum(r['original_wrong'] for r in selected)/w) if w else ''
                        entropy_errors[f'error{coverage}_pp'] = error
                k10, hits10, _ = counts[10]
                _, hits30, error30 = counts[30]
                result['table54'].append({**base, 'score': score,
                    'hidden_recall30_pct': formatted(100*hits30/h) if h else '',
                    'lift10': formatted((hits10/k10)/(h/n), 3) if h else '',
                    'error30_pp': formatted(error30, 3)})
            result['table52'].append({**base, **original_recalls})
            result['table53'].append({**base, 'strict_acc_pct': formatted(100*s/n),
                **{key: formatted(value, 3) for key, value in entropy_errors.items()}})
            errors.append({'strict_acc_pct': 100*s/n, **entropy_errors})
        if len(errors) == len(SIZES):
            result['table53'].append({'model': model, 'dataset': 'macro_average', 'N': '',
                **{key: formatted(sum(row[key] for row in errors)/len(errors),
                    2 if key == 'strict_acc_pct' else 3) for key in errors[0]}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=[ROOT/'data/predictions.csv.gz'], nargs='+')
    parser.add_argument('--output', type=Path, default=ROOT/'results')
    parser.add_argument('--check', action='store_true', help='Compare with packaged reference CSVs without writing')
    args = parser.parse_args()
    if args.input == [ROOT/'data/predictions.csv.gz']:
        manifest = json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
        if hashlib.sha256((ROOT/'data/predictions.csv.gz').read_bytes()).hexdigest() != manifest['predictions_sha256']:
            raise ValueError('Packaged predictions checksum mismatch')
    result = tables([row for path in args.input for row in read_records(path)])
    if not args.check:
        args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in result.items():
        if not rows:
            continue
        path = args.output/f'{name}.csv'
        if args.check:
            if read_records(path) != [{k: str(v) for k, v in row.items()} for row in rows]:
                raise ValueError(f'{path.name}: values differ')
        else:
            with path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        print(f'{name}: {len(rows)} rows' + (' verified' if args.check else ' written'))


if __name__ == '__main__':
    main()
