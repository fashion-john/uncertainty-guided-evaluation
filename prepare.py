"""Prepare only the four benchmarks and three views used in the paper."""
import argparse
import hashlib
import json
from pathlib import Path

from protocol import LABELS, MODES, SIZES, make_views, position_map

ROOT = Path(__file__).resolve().parent


def fingerprint(bench):
    return hashlib.sha256(json.dumps(bench, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def compact(bench, dataset):
    result = {mode: {} for mode in MODES}
    if len(bench['mcq']) != SIZES[dataset]:
        raise ValueError(f'{dataset}: expected {SIZES[dataset]} questions')
    for item_id, base in bench['mcq'].items():
        if list(base['options']) != list(LABELS):
            raise ValueError('Expected four ordered options A/B/C/D')
        expected = make_views(dataset, item_id, base)
        for mode in MODES:
            item = bench[mode][item_id]
            result[mode][item_id] = {key: item[key] for key in ('question', 'options', 'answer')}
            result[mode][item_id]['domain'] = item.get('domain', 'medical')
            for key in ('question', 'options', 'answer'):
                if item[key] != expected[mode][key]:
                    raise ValueError(f'{dataset}/{item_id}/{mode}: changed perturbation')
        # Force validation of the original identity mapping, including duplicate text.
        position_map(dataset, item_id, base)
    return result


def openbookqa():
    from datasets import load_dataset
    dataset = load_dataset('allenai/openbookqa', 'main', split='validation')
    bench = {mode: {} for mode in MODES}
    for index, row in enumerate(dataset):
        labels = [str(label).strip().upper() for label in row['choices']['label']]
        choices = [str(text).strip() for text in row['choices']['text']]
        if len(labels) != 4 or len(choices) != 4 or not all(choices):
            raise ValueError('OpenBookQA must have four options')
        base = {'question': str(row['question_stem']).strip(),
                'options': dict(zip(LABELS, choices)),
                'answer': LABELS[labels.index(str(row['answerKey']).strip().upper())]}
        item_id = f'openbookqa-{index:05d}'
        for mode, item in make_views('openbookqa', item_id, base).items():
            bench[mode][item_id] = item
    return bench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remedqa', type=Path, required=True, help='ReMedQA checkout containing data/bench/')
    parser.add_argument('--openbookqa', type=Path, help='Existing three-view OpenBookQA JSON (otherwise use Hugging Face)')
    parser.add_argument('--output', type=Path, default=ROOT/'data/benchmarks')
    args = parser.parse_args()
    expected = json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))['benchmarks']
    args.output.mkdir(parents=True, exist_ok=True)
    for name in SIZES:
        if name == 'openbookqa':
            bench = json.loads(args.openbookqa.read_text(encoding='utf-8')) if args.openbookqa else openbookqa()
        else:
            bench = json.loads((args.remedqa/f'data/bench/{name}_all.json').read_text(encoding='utf-8'))
        bench = compact(bench, name)
        if fingerprint(bench) != expected[name]['content_sha256']:
            raise ValueError(f'{name}: content differs from the published experiment')
        (args.output/f'{name}.json').write_text(json.dumps(bench, ensure_ascii=False), encoding='utf-8')
        print(f'{name}: {SIZES[name]} questions; content verified')


if __name__ == '__main__':
    main()
