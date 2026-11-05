"""Shared three-view protocol. Only Python's standard library is required."""
import math
import random
import re

LABELS = 'ABCD'
ROMAN = ('I', 'II', 'III', 'IV')
MODES = ('mcq', 'roman_numeral', 'fixed_pos')
SIZES = {'medqa': 1259, 'medmcqa': 1000, 'mmlu': 895, 'openbookqa': 500}
MODELS = ('Gemini-2.5-Flash', 'Qwen2.5-7B-Instruct', 'Gemma-3-12B-IT',
          'Llama3-Med42-8B', 'DeepSeek-V4-Flash', 'DeepSeek-V4-Pro')
CORE = MODELS[:4]
COVERAGES = (10, 20, 30, 40)
FIELDS = ('model', 'dataset', 'item_id', 'source_order', 'gold_mcq', 'gold_label',
          'gold_position', 'pred_mcq', 'pred_label', 'pred_position', 'position_map',
          'p_a', 'p_b', 'p_c', 'p_d', 'probability_imputed')


def parse_label(text, allowed=LABELS):
    text = (text or '').strip().upper()
    if 'FINAL ANSWER' in text:
        text = re.split(r'FINAL ANSWER\s*[:：]', text)[-1]
    found = {label for label in allowed if re.search(
        rf'(?<![A-Z0-9_]){re.escape(label)}(?![A-Z0-9_])', text)}
    return next(iter(found)) if len(found) == 1 else ''


def prompt(item):
    labels = list(item['options'])
    task = 'medical multiple-choice question' if item['domain'] == 'medical' else 'multiple-choice question'
    options = '\n'.join(f'{lab}. {item["options"][lab]}' for lab in labels)
    return (f'Answer the {task}. Return exactly one option label and nothing else. '
            'Do not include the option text, explanation, or punctuation.\n\n'
            f'Question: {item["question"]}\nOptions:\n{options}\n\n'
            f'Answer with exactly one label from: {", ".join(labels)}.\nAnswer:')


def position_map(dataset, item_id, base):
    permutation = list(LABELS)
    if dataset == 'openbookqa':
        rng = random.Random(f'fixed-pos-{item_id}')
        for _ in range(20):
            rng.shuffle(permutation)
            if permutation != list(LABELS):
                break
    elif base['answer'] != 'D':
        # Match the medical benchmark's historical first-matching-text swap.
        text = base['options'][base['answer']]
        first = next(i for i, lab in enumerate(LABELS) if base['options'][lab] == text)
        permutation[first], permutation[3] = permutation[3], permutation[first]
    return ''.join(permutation)  # displayed A/B/C/D -> original option identity


def make_views(dataset, item_id, base):
    domain = 'general' if dataset == 'openbookqa' else 'medical'
    base = {**base, 'domain': domain}
    mapping = position_map(dataset, item_id, base)
    roman = {**base, 'options': dict(zip(ROMAN, base['options'].values())),
             'answer': ROMAN[LABELS.index(base['answer'])]}
    fixed = {**base, 'options': {lab: base['options'][orig] for lab, orig in zip(LABELS, mapping)},
             'answer': LABELS[mapping.index(base['answer'])] if dataset == 'openbookqa' else 'D'}
    return dict(zip(MODES, (base, roman, fixed)))


def distribution(candidates, gemini=False):
    """Normalize label logprobs; retain the archived missing-label floor rule."""
    found, observed, seen = {}, [], set()
    for candidate in candidates:
        value = candidate.get('logProbability')
        if value is None or not math.isfinite(float(value)):
            continue
        token = str(candidate.get('token', ''))
        identity = candidate.get('tokenId', token)
        if identity in seen:
            continue
        seen.add(identity)
        observed.append(float(value))
        label = token.strip() if gemini else token.replace('Ġ', '').replace('▁', '').strip().strip('()[]')
        if label in LABELS:
            if label in found and not gemini:
                peak = max(found[label], float(value))
                found[label] = peak + math.log(math.exp(found[label]-peak) + math.exp(float(value)-peak))
            else:
                found[label] = float(value)
    if not found:
        return {}, False
    missing = set(LABELS) - found.keys()
    floor = min(observed) - 5
    logps = [found.get(lab, floor) for lab in LABELS]
    weights = [math.exp(value-max(logps)) for value in logps]
    return dict(zip(LABELS, (w/sum(weights) for w in weights))), bool(missing)


def record(model, dataset, item_id, order, views, texts, probabilities, imputed):
    return dict(zip(FIELDS, (model, dataset, item_id, order,
        *(views[mode]['answer'] for mode in MODES),
        *(parse_label(texts[mode], views[mode]['options']) for mode in MODES),
        position_map(dataset, item_id, views['mcq']),
        *(probabilities.get(lab, '') for lab in LABELS), int(imputed))))


def measurements(row):
    """Rebuild correctness and consistency from labels, not saved metric columns."""
    dataset = row['dataset']
    if dataset not in SIZES or row['model'] not in MODELS:
        raise ValueError('Unknown model or dataset')
    allowed = (LABELS, ROMAN, LABELS)
    gold = [row['gold_'+name] for name in ('mcq', 'label', 'position')]
    predicted = [row['pred_'+name] for name in ('mcq', 'label', 'position')]
    if any(g not in space or (p and p not in space) for g, p, space in zip(gold, predicted, allowed)):
        raise ValueError('Invalid gold/predicted label')
    mapping = row['position_map']
    if len(mapping) != 4 or set(mapping) != set(LABELS):
        raise ValueError('Invalid position mapping')
    correct = [g == p for g, p in zip(gold, predicted)]
    identities = [predicted[0], LABELS[ROMAN.index(predicted[1])] if predicted[1] else '',
                  mapping[LABELS.index(predicted[2])] if predicted[2] else '']
    probabilities = [row['p_'+lab.lower()] for lab in LABELS]
    scores = {}
    if any(value != '' for value in probabilities):
        p = [float(value) for value in probabilities]
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in p) or abs(sum(p)-1) > 1e-10:
            raise ValueError('Invalid candidate probabilities')
        ranked = sorted(p, reverse=True)
        scores = {'Entropy': -sum(v*math.log(v+1e-15) for v in p),
                  'Margin': ranked[0]-ranked[1], 'Max-prob': ranked[0]}
    c, label, position = correct
    return {**row, 'source_order': int(row['source_order']), 'correct': int(c),
            'strict': int(c and label and position), 'original_wrong': int(not c),
            'hidden': int(c and not (label and position)), 'label_pair': int(c and label),
            'position_pair': int(c and position),
            'consistent': int(all(identities) and len(set(identities)) == 1), 'scores': scores}
