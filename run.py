"""Run the original, label-renamed and position-changed views of one benchmark."""
import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from prepare import ROOT, compact, fingerprint
from protocol import CORE, FIELDS, LABELS, MODELS, MODES, ROMAN, SIZES, distribution, prompt, record


class LocalModel:
    def __init__(self, args):
        from vllm import LLM, SamplingParams
        self.params = SamplingParams
        checkpoint = args.model
        config_path = Path(checkpoint)/'config.json'
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding='utf-8'))
            quant = config.get('quantization_config')
            empty = isinstance(quant, dict) and not quant.get('quant_method', quant.get('quantization_method', quant.get('load_in')))
            if empty or (isinstance(quant, str) and not quant.strip()):
                # Some Med42 snapshots have empty quantization metadata. Preserve
                # the downloaded weights/config and correct only a temporary copy.
                config.pop('quantization_config', None)
                if isinstance(config.get('quantization'), str) and not config['quantization'].strip():
                    config.pop('quantization')
                self.shadow = tempfile.TemporaryDirectory(prefix='llm-evaluation-')
                checkpoint = self.shadow.name
                for child in config_path.parent.iterdir():
                    if child.name != 'config.json':
                        (Path(checkpoint)/child.name).symlink_to(child.resolve(), target_is_directory=child.is_dir())
                (Path(checkpoint)/'config.json').write_text(json.dumps(config), encoding='utf-8')
        self.model = LLM(model=checkpoint, dtype='bfloat16', tensor_parallel_size=args.tensor_parallel,
                         gpu_memory_utilization=args.gpu_memory, max_model_len=4096, trust_remote_code=True)
        self.tokenizer = self.model.get_tokenizer()

    def generate(self, items, mode):
        labels = ROMAN if mode == 'roman_numeral' else LABELS
        tokens = {}
        for label in labels:
            tokens[label] = set()
            for variant in (label, ' '+label, '('+label, ' ('+label):
                encoded = self.tokenizer.encode(variant, add_special_tokens=False)
                if len(encoded) == 1:
                    tokens[label].add(encoded[0])
        if any(not ids for ids in tokens.values()):
            raise ValueError('Each constrained answer label must have a single-token encoding')
        prompts = []
        for item in items:
            text = prompt(item)
            if getattr(self.tokenizer, 'chat_template', None):
                text = self.tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': text}], tokenize=False, add_generation_prompt=True)
            prompts.append(text)
        params = self.params(temperature=0, max_tokens=1, logprobs=20,
                            allowed_token_ids=sorted(set.union(*tokens.values())))
        outputs = self.model.generate(prompts, params, use_tqdm=False)
        result = []
        for output in outputs:
            completion = output.outputs[0]
            first = completion.logprobs[0] if completion.logprobs else {}
            candidates = [{'token': value.decoded_token if value.decoded_token is not None
                           else self.tokenizer.decode([key]), 'tokenId': key, 'logProbability': value.logprob}
                          for key, value in first.items()]
            result.append((completion.text, candidates))
        return result


class RemoteModel:
    def __init__(self, args):
        self.args = args
        self.key = os.environ.get(args.api_key_env)
        if not self.key:
            raise ValueError(f'Set the {args.api_key_env} environment variable')
        self.base = args.api_base.rstrip('/')

    def generate(self, items, mode):
        return [self.one(item, mode) for item in items]

    def one(self, item, mode):
        args = self.args
        want_probs = mode == 'mcq' and args.model_name in CORE
        if args.backend == 'gemini':
            url = f'{self.base}/v1beta/models/{args.model}:generateContent'
            config = {'temperature': 0, 'maxOutputTokens': 8 if mode == 'mcq' else 96,
                      'thinkingConfig': {'thinkingBudget': 0}}
            if want_probs:
                config.update(responseLogprobs=True, logprobs=20)
            body = {'contents': [{'parts': [{'text': prompt(item)}]}], 'generationConfig': config}
            headers = {'x-goog-api-key': self.key}
        else:
            url = f'{self.base}/chat/completions'
            body = {'model': args.model, 'messages': [{'role': 'user', 'content': prompt(item)}],
                    'temperature': 0, 'max_tokens': 8}
            if args.model_name.startswith('DeepSeek'):
                body['thinking'] = {'type': 'disabled'}
            if want_probs:
                body.update(logprobs=True, top_logprobs=20)
            headers = {'Authorization': 'Bearer '+self.key}
        request = urllib.request.Request(url, json.dumps(body).encode(),
                                         {**headers, 'Content-Type': 'application/json'})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = json.load(response)
                break
            except urllib.error.HTTPError as error:
                if error.code not in (408, 429, 500, 502, 503, 504) or attempt == 3:
                    # Do not print server error bodies or request headers containing credentials.
                    raise RuntimeError(f'Model request failed (HTTP {error.code})') from None
            except (urllib.error.URLError, TimeoutError):
                if attempt == 3:
                    raise RuntimeError('Model request failed after four attempts') from None
            time.sleep(1.5*(attempt+1))
        if args.backend == 'gemini':
            first = (data.get('candidates') or [{}])[0]
            text = ''.join(part.get('text', '') for part in first.get('content', {}).get('parts', []))
            top = first.get('logprobsResult', {}).get('topCandidates') or [{}]
            return text, top[0].get('candidates', [])
        if not data.get('choices'):
            raise RuntimeError('Model response has no choices')
        first = data['choices'][0]
        steps = (first.get('logprobs') or {}).get('content') or [{}]
        top = [{'token': c['token'], 'logProbability': c['logprob']}
               for c in steps[0].get('top_logprobs', [])]
        return first.get('message', {}).get('content') or '', top


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('vllm', 'gemini', 'openai'), required=True)
    parser.add_argument('--model', required=True, help='Local checkpoint path or provider model ID')
    parser.add_argument('--model-name', choices=MODELS, required=True, help='Name used in the result tables')
    parser.add_argument('--dataset', choices=SIZES, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Per-question CSV; existing completed rows are resumed')
    parser.add_argument('--api-base', default='', help='Native Gemini host or OpenAI-compatible base URL')
    parser.add_argument('--api-key-env', default='MODEL_API_KEY')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--tensor-parallel', type=int, default=1)
    parser.add_argument('--gpu-memory', type=float, default=0.90)
    parser.add_argument('--limit', type=int, help='Optional smoke run; incomplete datasets cannot reproduce paper tables')
    parser.add_argument('--dry-run', action='store_true', help='Validate data and show one prompt without loading/calling a model')
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error('batch-size and limit must be positive')
    if not args.api_base:
        args.api_base = 'https://generativelanguage.googleapis.com' if args.backend == 'gemini' else 'https://api.deepseek.com'
    bench = compact(json.loads(args.data.read_text(encoding='utf-8')), args.dataset)
    expected = json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))['benchmarks'][args.dataset]
    if fingerprint(bench) != expected['content_sha256']:
        raise ValueError('Benchmark content differs from the published experiment')
    ids = sorted(bench['mcq'])
    if args.dry_run:
        print(f'{args.dataset}: {len(ids)} questions, {3*len(ids)} model calls for full evaluation')
        print(prompt(bench['mcq'][ids[0]]))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'dataset': args.dataset, 'model_name': args.model_name, 'model': args.model,
                'backend': args.backend, 'bench_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
                'temperature': 0, 'max_model_len': 4096 if args.backend == 'vllm' else None,
                'constrained_labels': args.backend == 'vllm'}
    sidecar = args.output.with_suffix('.json')
    done = set()
    if args.output.exists():
        if not sidecar.exists() or json.loads(sidecar.read_text(encoding='utf-8')) != metadata:
            raise ValueError('Output belongs to a different model or benchmark')
        with args.output.open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(FIELDS):
                raise ValueError('Existing output has an incompatible schema')
            for row in reader:
                if row['item_id'] in done or row['model'] != args.model_name or row['dataset'] != args.dataset:
                    raise ValueError('Existing output contains inconsistent rows')
                done.add(row['item_id'])
    sidecar.write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    pending = [(order, item_id) for order, item_id in enumerate(ids[:args.limit]) if item_id not in done]
    if not pending:
        print('All selected questions are already complete')
        return
    backend = LocalModel(args) if args.backend == 'vllm' else RemoteModel(args)
    with args.output.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start+args.batch_size]
            responses = {mode: backend.generate([bench[mode][item_id] for _, item_id in batch], mode) for mode in MODES}
            for i, (order, item_id) in enumerate(batch):
                probabilities, imputed = distribution(responses['mcq'][i][1], gemini=args.backend == 'gemini')
                if args.model_name in CORE and not probabilities:
                    raise ValueError('Missing original candidate probabilities; cannot evaluate risk ranking')
                if args.model_name not in CORE:
                    probabilities, imputed = {}, False
                writer.writerow(record(args.model_name, args.dataset, item_id, order,
                    {mode: bench[mode][item_id] for mode in MODES},
                    {mode: responses[mode][i][0] for mode in MODES}, probabilities, imputed))
                stream.flush()
            print(f'{min(start+len(batch), len(pending))}/{len(pending)} new questions completed', flush=True)


if __name__ == '__main__':
    main()
