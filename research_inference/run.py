"""Registered inference and isolated recorded-input replay; preserves history."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mnl_inference import from_swissmetro, fit, covariance, inference_rows


def hashes(paths):
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}


def acquire_input(path, receipt):
    if path is None:
        with urllib.request.urlopen(receipt['source_url'], timeout=60) as response:
            raw = response.read()
    else:
        raw = Path(path).read_bytes()
    if len(raw) != receipt['raw_bytes'] or hashlib.sha256(raw).hexdigest() != receipt['raw_sha256']:
        raise ValueError('input bytes differ from registered official source')
    frame = pd.read_csv(io.BytesIO(raw), sep='\t')
    sample = frame.loc[frame.PURPOSE.isin([1, 3]) & frame.CHOICE.ne(0)].copy()
    if len(frame) != receipt['raw_rows'] or len(sample) != receipt['usable_rows'] or sample.ID.nunique() != receipt['respondents']:
        raise ValueError('registered row/respondent identities do not match')
    return raw, sample


def isolated_legacy_replay(raw):
    """Replay unchanged computational source; HTTP acquisition is not tested.

    The original requests loader receives the hash-verified registered bytes,
    and every file it writes stays in a temporary directory. No fitted model,
    split, threshold, output schema or source expression is patched.
    """
    spec = importlib.util.spec_from_file_location('historical_choice_pipeline', ROOT/'src/discrete_choice_pipeline.py')
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    old_get = core.requests.get
    class RecordedResponse:
        content = raw
        @staticmethod
        def raise_for_status():
            return None
    def recorded_get(url, timeout=None):
        if url != core.DATA_URL:
            raise ValueError('historical loader requested an unexpected source')
        return RecordedResponse()
    old_directory = Path.cwd()
    captured = io.StringIO()
    comparisons = {}
    try:
        with tempfile.TemporaryDirectory(prefix='choice-replay-') as temporary:
            core.requests.get = recorded_get
            os.chdir(temporary)
            with contextlib.redirect_stdout(captured):
                core.main()
            for name in ['respondent_holdout_validation.csv', 'swissmetro_pricing_scenarios.csv', 'project7_metrics.csv']:
                prior = pd.read_csv(ROOT/'results'/name)
                replay = pd.read_csv(Path(temporary)/'results'/name)
                if list(prior.columns) != list(replay.columns) or prior.shape != replay.shape:
                    raise ValueError(f'legacy output schema changed: {name}')
                maximum_delta = 0.0
                for column in prior.columns:
                    if pd.api.types.is_numeric_dtype(prior[column]):
                        if not np.allclose(prior[column], replay[column], atol=1e-6, rtol=1e-6, equal_nan=False):
                            raise ValueError(f'legacy numeric mismatch: {name}/{column}')
                        maximum_delta = max(maximum_delta, float(np.max(np.abs(prior[column]-replay[column]))))
                    elif not prior[column].equals(replay[column]):
                        raise ValueError(f'legacy categorical mismatch: {name}/{column}')
                comparisons[name] = {'schema_equal': True, 'numeric_tolerance_pass': True, 'maximum_absolute_delta': maximum_delta}
            estimates = pd.read_csv(Path(temporary)/'results/mnl_point_estimates_regenerated.csv')
            prior_estimates = pd.read_csv(ROOT/'results/mnl_parameter_estimates.csv')
            if not estimates.parameter.equals(prior_estimates.parameter) or not np.allclose(estimates.estimate, prior_estimates.estimate, atol=1e-6, rtol=1e-6):
                raise ValueError('legacy point estimates differ from historical table')
    finally:
        core.requests.get = old_get
        os.chdir(old_directory)
    return {'scope': 'unchanged core computational replay from registered source bytes; historical HTTP acquisition not tested',
            'historical_inference_columns_regenerated': False,
            'csv_comparisons': comparisons, 'point_estimates_match': True,
            'stdout': captured.getvalue()}


def serializable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, dict): return {k: serializable(v) for k,v in value.items()}
    if isinstance(value, list): return [serializable(v) for v in value]
    if isinstance(value, np.generic): return value.item()
    return value


def run(data_path, output):
    receipt = json.loads((ROOT/'research_inference/INPUT_SOURCE.json').read_text())
    expected = json.loads((ROOT/'research_inference/HISTORICAL_SHA256.json').read_text())
    before = hashes(expected)
    if before != expected:
        raise ValueError('historical source or result bytes differ from the frozen inventory')
    output = Path(output).resolve()
    if output == ROOT.resolve() or (ROOT/'results').resolve() == output or (ROOT/'results').resolve() in output.parents:
        raise ValueError('new output must be separate from committed historical results')
    if output.exists():
        raise FileExistsError('refusing to overwrite any existing output path')
    raw, sample = acquire_input(data_path, receipt)
    data = from_swissmetro(sample)
    beta, optimizer = fit(data)
    estimates = covariance(beta, data)
    if abs(estimates['log_likelihood']-receipt['official_benchmark']) >= .05:
        raise ValueError('official full-sample likelihood benchmark not reproduced')
    rows = inference_rows(beta, estimates)
    replay = isolated_legacy_replay(raw)
    after = hashes(expected)
    if before != after:
        raise ValueError('historical byte preservation failed')
    proof = {'source': receipt, 'optimizer': optimizer, 'inference': serializable(estimates),
             'parameter_table': rows, 'legacy_replay': replay,
             'historical_sha256': after, 'historical_bytes_preserved': True,
             'scope': 'same known stated-preference sample and fixed MNL specification; asymptotic respondent-cluster inference, not new holdout, causal fare effect, population coverage or real-market validation'}
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(output/'parameter_inference.csv', index=False)
    (output/'summary.json').write_text(json.dumps(proof, indent=2, allow_nan=False)+'\n')
    print('INFERENCE_RESULT_JSON=' + json.dumps(proof, allow_nan=False, separators=(',', ':')))
    print(json.dumps({'rows': estimates['rows'], 'respondents': estimates['clusters'],
                      'log_likelihood': estimates['log_likelihood'], 'gradient_infinity_norm': optimizer['gradient_infinity_norm'],
                      'historical_bytes_preserved': True, 'legacy_point_and_metric_replay': True,
                      'output': str(output)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, help='optional local input, verified against the registered SHA256')
    parser.add_argument('--output', type=Path, required=True, help='new, nonexistent output directory outside historical results')
    args = parser.parse_args()
    run(args.data, args.output)
