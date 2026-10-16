#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


TIMEOUT_SEC = 20 * 60
TIMEOUT_MS = TIMEOUT_SEC * 1000


def parse_args():
    bundle_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description='Run default Choco and Rank-priority Choco on a saved split.')
    parser.add_argument('--train-dir', required=True,
                        help='Rank train_dir containing metadata, numpy_weights.npz, and split_instances.json.')
    parser.add_argument('--split-file', default=None,
                        help='Split JSON. Defaults to <train-dir>/split_instances.json.')
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'],
                        help='Which split to evaluate.')
    parser.add_argument('--instances-dir', default=None,
                        help='Directory used to resolve relative instance names. Searched recursively by basename.')
    parser.add_argument('--output-dir', default=str(bundle_dir / 'test_walltimes'),
                        help='Directory for the walltime CSV.')
    parser.add_argument('--log-file', default=None,
                        help='CSV log path. Defaults to <output-dir>/<split>_walltimes.csv.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Evaluate at most this many instances.')
    parser.add_argument('--jobs', type=int, default=1,
                        help='Number of instances to evaluate in parallel. Each job runs default then model.')
    parser.add_argument('--command-timeout-sec', type=float, default=TIMEOUT_SEC,
                        help='Outer timeout for each Java process. Defaults to 1200 seconds.')
    parser.add_argument('--rank-jar',
                        default=str(bundle_dir / 'choco/dist/choco-rank.jar'))
    parser.add_argument('--rank-dir', default=str(bundle_dir),
                        help='Project root containing src/choco_priority_pipeline.py.')
    parser.add_argument('--feature-extractor',
                        default=str(bundle_dir / 'feature_extractor/xcsp2feat'))
    parser.add_argument('--rank-python', default=None,
                        help='Python executable used by the Java Rank runner.')
    parser.add_argument('--gated-policy', action='store_true',
                        help='Run the model side with choco-rank.jar --run-gated-model.')
    parser.add_argument('--class-policy', default=None,
                        help='Class policy JSON. Defaults to <train-dir>/class_policy.json when --gated-policy is used.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Resolve instances and print what would run without launching Java.')
    return parser.parse_args()


def load_split_records(split_file, split_name):
    with open(split_file) as fin:
        records = json.load(fin)
    selected = [record for record in records if record.get('split') == split_name]
    selected.sort(key=lambda row: (row.get('problem_class', ''), row.get('instance', '')))
    return selected


def require_path(path, description, executable=False):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError('%s not found: %s' % (description, path))
    if executable and not os.access(str(path), os.X_OK):
        raise PermissionError('%s is not executable: %s' % (description, path))
    return path


def validate_runtime_inputs(args, train_dir, split_file):
    require_path(train_dir, 'train-dir')
    require_path(split_file, 'split file')
    require_path(train_dir / 'metadata.json', 'model metadata.json')
    require_path(train_dir / 'numpy_weights.npz', 'model numpy_weights.npz')
    require_path(args.rank_jar, 'Rank Choco jar')
    rank_dir = require_path(args.rank_dir, 'Rank project directory')
    require_path(rank_dir / 'src/choco_priority_pipeline.py',
                 'Rank priority pipeline')
    require_path(rank_dir / 'src/predict_priorities.py',
                 'NumPy priority predictor')
    require_path(args.feature_extractor, 'feature extractor', executable=True)
    if args.class_policy is not None:
        require_path(args.class_policy, 'class policy')
    if args.rank_python is not None:
        require_path(args.rank_python, 'rank python', executable=True)


def build_instance_index(instances_dir):
    if instances_dir is None:
        return {}
    root = Path(instances_dir).resolve()
    if not root.is_dir():
        raise ValueError('instances-dir does not exist: %s' % root)

    index = {}
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('.xml'):
                index.setdefault(filename, Path(dirpath) / filename)
    return index


def resolve_instance(instance_value, instances_dir, instance_index):
    candidate = Path(instance_value)
    if candidate.is_file():
        return candidate.resolve()

    if instances_dir is not None:
        joined = Path(instances_dir) / instance_value
        if joined.is_file():
            return joined.resolve()

    basename = Path(instance_value).name
    if basename in instance_index:
        return instance_index[basename].resolve()

    raise FileNotFoundError('Could not resolve instance: %s' % instance_value)


def safe_stem(path):
    stem = Path(path).name
    if stem.endswith('.xml'):
        stem = stem[:-4]
    return ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in stem)


def run_command(command, cwd, timeout_sec, extra_env=None):
    start = time.monotonic()
    status = 'ok'
    return_code = 0
    output = ''
    def text_output(value):
        if value is None:
            return ''
        if isinstance(value, bytes):
            return value.decode('utf-8', errors='replace')
        return value

    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=merged_env(extra_env),
            timeout=timeout_sec)
        return_code = completed.returncode
        output = text_output(completed.stdout)
        if return_code != 0:
            status = 'error'
    except subprocess.TimeoutExpired as exc:
        status = 'timeout'
        return_code = -1
        output = text_output(exc.stdout)
        if exc.stderr:
            output += text_output(exc.stderr)
    walltime_ms = int(round((time.monotonic() - start) * 1000.0))
    if status == 'timeout':
        walltime_ms = int(round(timeout_sec * 1000.0))

    return status, return_code, walltime_ms


def merged_env(extra_env):
    if not extra_env:
        return None
    env = os.environ.copy()
    env.update(extra_env)
    return env


def read_choco_runtime_ms(result_json, status=None, timeout_ms=TIMEOUT_MS):
    if not result_json.is_file():
        if status == 'timeout':
            return timeout_ms
        return ''
    try:
        with open(result_json) as fin:
            rows = json.load(fin)
        if rows:
            runtime_ms = rows[0].get('runtime_ms', '')
            if runtime_ms != '':
                return runtime_ms
    except (OSError, ValueError, TypeError):
        if status == 'timeout':
            return timeout_ms
        return ''
    if status == 'timeout':
        return timeout_ms
    return ''


def evaluate_instance(task):
    index = task['index']
    total = task['total']
    record = task['record']
    instance_path = Path(task['instance_path'])
    bundle_dir = Path(task['bundle_dir'])
    train_dir = Path(task['train_dir'])
    timeout_sec = task['command_timeout_sec']
    timeout_ms = int(round(timeout_sec * 1000.0))

    stem = safe_stem(instance_path)
    with tempfile.TemporaryDirectory(prefix='chocorank_eval_') as temp_dir:
        temp_root = Path(temp_dir)
        default_result = temp_root / (stem + '.default.json')
        model_result = temp_root / (stem + '.model.json')

        default_command = [
            'java', '-jar', str(Path(task['rank_jar']).resolve()),
            '--run-default', str(default_result), str(instance_path)]
        default_status, default_code, default_wall_ms = run_command(
            default_command, bundle_dir, timeout_sec)

        model_mode = '--run-gated-model' if task['gated_policy'] else '--run-model'
        model_command = [
            'java', '-jar', str(Path(task['rank_jar']).resolve()),
            model_mode,
            str(model_result),
            str(Path(task['rank_dir']).resolve()),
            str(train_dir),
            str(Path(task['feature_extractor']).resolve()),
            str(instance_path)]
        if task['gated_policy'] and task['class_policy'] is not None:
            model_command.append(str(Path(task['class_policy']).resolve()))
        model_env = {}
        if task['rank_python'] is not None:
            model_env['RANK_PYTHON'] = str(Path(task['rank_python']).resolve())
        model_status, model_code, model_wall_ms = run_command(
            model_command, bundle_dir, timeout_sec, model_env)

        row = {
            'instance': record.get('instance', instance_path.name),
            'default_walltime_ms': default_wall_ms,
            'model_walltime_ms': model_wall_ms,
            'default_choco_runtime_ms': read_choco_runtime_ms(
                default_result, default_status, timeout_ms),
            'model_choco_runtime_ms': read_choco_runtime_ms(
                model_result, model_status, timeout_ms),
        }
    return {
        'index': index,
        'total': total,
        'instance': instance_path.name,
        'default_status': default_status,
        'default_code': default_code,
        'model_status': model_status,
        'model_code': model_code,
        'row': row,
    }


def build_tasks(args, bundle_dir, train_dir, resolved):
    total = len(resolved)
    tasks = []
    for index, (record, instance_path) in enumerate(resolved, start=1):
        tasks.append({
            'index': index,
            'total': total,
            'record': record,
            'instance_path': str(instance_path),
            'bundle_dir': str(bundle_dir),
            'train_dir': str(train_dir),
            'command_timeout_sec': args.command_timeout_sec,
            'rank_jar': args.rank_jar,
            'rank_dir': args.rank_dir,
            'feature_extractor': args.feature_extractor,
            'rank_python': args.rank_python,
            'gated_policy': args.gated_policy,
            'class_policy': args.class_policy,
        })
    return tasks


def main():
    args = parse_args()
    bundle_dir = Path(__file__).resolve().parent.parent
    train_dir = Path(args.train_dir).resolve()
    split_file = Path(args.split_file).resolve() if args.split_file else train_dir / 'split_instances.json'
    output_dir = Path(args.output_dir).resolve()
    log_file = Path(args.log_file).resolve() if args.log_file else output_dir / ('%s_walltimes.csv' % args.split)

    validate_runtime_inputs(args, train_dir, split_file)
    records = load_split_records(split_file, args.split)
    if args.limit is not None:
        records = records[:args.limit]
    instance_index = build_instance_index(args.instances_dir)

    resolved = []
    for record in records:
        instance_path = resolve_instance(record['instance'], args.instances_dir, instance_index)
        resolved.append((record, instance_path))

    if args.dry_run:
        print('split=%s instances=%d' % (args.split, len(resolved)))
        for record, instance_path in resolved:
            print('%s\t%s' % (record.get('problem_class', ''), instance_path))
        return

    log_file.parent.mkdir(parents=True, exist_ok=True)
    jobs = max(1, int(args.jobs))

    fieldnames = [
        'instance',
        'default_walltime_ms',
        'model_walltime_ms',
        'default_choco_runtime_ms',
        'model_choco_runtime_ms',
    ]

    with open(log_file, 'w', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        tasks = build_tasks(
            args, bundle_dir, train_dir, resolved)
        if jobs == 1:
            for task in tasks:
                print('[%d/%d] default+model %s' %
                      (task['index'], task['total'], Path(task['instance_path']).name))
                result = evaluate_instance(task)
                writer.writerow(result['row'])
                fout.flush()
        else:
            print('Running %d instances with %d parallel jobs' % (len(tasks), jobs))
            with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
                futures = [executor.submit(evaluate_instance, task) for task in tasks]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    print('[%d/%d] done %s default=%s model=%s' %
                          (result['index'], result['total'], result['instance'],
                           result['default_status'], result['model_status']))
                    writer.writerow(result['row'])
                    fout.flush()

    print('Wrote %s' % log_file)


if __name__ == '__main__':
    main()
