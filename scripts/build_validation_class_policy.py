#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


TIMEOUT_SEC = 20 * 60
TIMEOUT_MS = TIMEOUT_SEC * 1000


def parse_args():
    bundle_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            'Evaluate default Choco vs a Rank model on validation instances '
            'and write a per-problem-class model/default policy.'))
    parser.add_argument('--train-dir', required=True,
                        help='Rank train_dir containing metadata and split_instances files.')
    parser.add_argument('--instances-dir', required=True,
                        help='Directory used to resolve validation XML instances.')
    parser.add_argument('--walltimes-csv', default=None,
                        help='Use an existing validation walltime CSV instead of running evaluation.')
    parser.add_argument('--policy-output', default=None,
                        help='Policy JSON path. Defaults to <train-dir>/class_policy.json.')
    parser.add_argument('--split-file', default=None,
                        help='Split JSON/CSV base. Defaults to <train-dir>/split_instances.json.')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--command-timeout-sec', type=float, default=TIMEOUT_SEC)
    parser.add_argument('--par', type=float, default=1.0,
                        help='PAR multiplier used for walltime policy aggregation. Default: PAR1.')
    parser.add_argument('--min-improvement', type=float, default=0.0,
                        help='Minimum relative PAR walltime improvement required to choose model, e.g. 0.05 for 5%%.')
    parser.add_argument('--allow-fewer-solved', action='store_true',
                        help='Allow model policy even when it solves fewer validation instances.')
    parser.add_argument('--min-validation-instances', type=int, default=1,
                        help='Minimum validation instances needed to choose model by measured stats.')
    parser.add_argument('--small-class-decision',
                        choices=['default', 'model'],
                        default='default',
                        help='Decision for classes with fewer than --min-validation-instances.')
    parser.add_argument('--unseen-class-decision',
                        choices=['default', 'model'],
                        default='model',
                        help='Runtime fallback for problem classes unseen in training/policy.')
    parser.add_argument('--rank-python', default=None,
                        help='Forwarded to evaluate_test_walltimes.py.')
    return parser.parse_args()


def parse_ms(value):
    value = (value or '').strip()
    if value == '':
        return None
    return float(value)


def solved(value, timeout_ms):
    return value is not None and value < timeout_ms


def penalized(value, timeout_ms, par_multiplier):
    if value is None or value >= timeout_ms:
        return par_multiplier * timeout_ms
    return value


def relative_improvement(default_value, model_value):
    if default_value <= 0:
        return 0.0
    return (default_value - model_value) / default_value


def load_split_records(split_path):
    split_path = Path(split_path)
    if split_path.suffix == '.csv':
        with split_path.open(newline='') as fin:
            return list(csv.DictReader(fin))
    with split_path.open() as fin:
        return json.load(fin)


def split_record_maps(train_dir, split_file):
    if split_file is None:
        split_file = train_dir / 'split_instances.json'
    records = load_split_records(split_file)
    class_by_instance = {}
    train_classes = set()
    valid_classes = set()
    for record in records:
        instance = record['instance']
        problem_class = record.get('problem_class') or infer_problem_class(instance)
        class_by_instance[instance] = problem_class
        if record.get('split') == 'train':
            train_classes.add(problem_class)
        if record.get('split') == 'valid':
            valid_classes.add(problem_class)
    return class_by_instance, train_classes, valid_classes


def infer_problem_class(instance_name):
    base = Path(instance_name).name
    if base.endswith('.xml'):
        base = base[:-4]
    if '-' in base:
        return base.split('-', 1)[0]
    return base


def run_validation_evaluation(args, train_dir, output_dir, walltimes_csv):
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / 'evaluate_test_walltimes.py'),
        '--train-dir', str(train_dir),
        '--instances-dir', str(Path(args.instances_dir).resolve()),
        '--output-dir', str(output_dir),
        '--log-file', str(walltimes_csv),
        '--split', 'valid',
        '--jobs', str(args.jobs),
        '--command-timeout-sec', str(args.command_timeout_sec),
    ]
    if args.split_file is not None:
        command.extend(['--split-file', str(Path(args.split_file).resolve())])
    if args.rank_python is not None:
        command.extend(['--rank-python', args.rank_python])

    completed = subprocess.run(
        command,
        cwd=str(Path(__file__).resolve().parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            'validation evaluation failed with exit code %d:\n%s' %
            (completed.returncode, completed.stdout[-4000:]))


def read_walltime_rows(csv_path, class_by_instance):
    rows = []
    unknown_instances = []
    with Path(csv_path).open(newline='') as fin:
        for row in csv.DictReader(fin):
            instance = row['instance']
            problem_class = class_by_instance.get(instance)
            if problem_class is None:
                problem_class = infer_problem_class(instance)
                unknown_instances.append(instance)
            rows.append({
                'instance': instance,
                'problem_class': problem_class,
                'default_walltime_ms': parse_ms(row.get('default_walltime_ms')),
                'model_walltime_ms': parse_ms(row.get('model_walltime_ms')),
                'default_solve_ms': parse_ms(row.get('default_choco_runtime_ms')),
                'model_solve_ms': parse_ms(row.get('model_choco_runtime_ms')),
            })
    return rows, unknown_instances


def class_stats(rows, timeout_ms, par_multiplier):
    stats = {}
    by_class = defaultdict(list)
    for row in rows:
        by_class[row['problem_class']].append(row)

    for problem_class, class_rows in sorted(by_class.items()):
        default_solve_sum = sum(
            penalized(row['default_solve_ms'], timeout_ms, par_multiplier)
            for row in class_rows)
        model_solve_sum = sum(
            penalized(row['model_solve_ms'], timeout_ms, par_multiplier)
            for row in class_rows)
        default_wall_sum = sum(
            penalized(row['default_walltime_ms'], timeout_ms, par_multiplier)
            for row in class_rows)
        model_wall_sum = sum(
            penalized(row['model_walltime_ms'], timeout_ms, par_multiplier)
            for row in class_rows)
        default_solved = sum(solved(row['default_solve_ms'], timeout_ms)
                             for row in class_rows)
        model_solved = sum(solved(row['model_solve_ms'], timeout_ms)
                           for row in class_rows)
        stats[problem_class] = {
            'validation_instances': len(class_rows),
            'default_solved': default_solved,
            'model_solved': model_solved,
            'solved_delta': model_solved - default_solved,
            'default_solve_par_ms': default_solve_sum,
            'model_solve_par_ms': model_solve_sum,
            'solve_par_improvement': relative_improvement(
                default_solve_sum, model_solve_sum),
            'default_walltime_par_ms': default_wall_sum,
            'model_walltime_par_ms': model_wall_sum,
            'walltime_par_improvement': relative_improvement(
                default_wall_sum, model_wall_sum),
        }
    return stats


def build_policy(stats, train_classes, valid_classes, args):
    class_policy = {}
    summary = defaultdict(int)
    all_classes = sorted(set(stats) | set(train_classes) | set(valid_classes))
    for problem_class in all_classes:
        class_seen_in_training = problem_class in train_classes
        class_stat = stats.get(problem_class)
        if not class_seen_in_training:
            decision = args.unseen_class_decision
            reason = 'unseen_in_training'
        elif class_stat is None:
            decision = 'default'
            reason = 'no_validation_measurement'
        elif class_stat['validation_instances'] < args.min_validation_instances:
            decision = args.small_class_decision
            reason = 'too_few_validation_instances'
        else:
            solved_ok = (
                args.allow_fewer_solved or
                class_stat['model_solved'] >= class_stat['default_solved'])
            improvement_ok = (
                class_stat['walltime_par_improvement'] >= args.min_improvement)
            if solved_ok and improvement_ok:
                decision = 'model'
                reason = 'validation_model_better'
            else:
                decision = 'default'
                reason = 'validation_default_better'

        entry = {
            'decision': decision,
            'use_model': decision == 'model',
            'reason': reason,
            'seen_in_training': class_seen_in_training,
        }
        if class_stat is not None:
            entry.update(class_stat)
        class_policy[problem_class] = entry
        summary[decision] += 1
        summary[reason] += 1
    return class_policy, dict(summary)


def write_policy(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as fout:
        json.dump(payload, fout, indent=2, sort_keys=True)
    print('Wrote %s' % path)


def main():
    args = parse_args()
    train_dir = Path(args.train_dir).resolve()
    policy_output = Path(args.policy_output).resolve() if args.policy_output else (
        train_dir / 'class_policy.json')
    split_file = Path(args.split_file).resolve() if args.split_file else None

    temp_context = None
    if args.walltimes_csv is None:
        temp_context = tempfile.TemporaryDirectory(prefix='chocorank_policy_')
        output_dir = Path(temp_context.name)
        walltimes_csv = output_dir / 'valid_walltimes.csv'
        run_validation_evaluation(args, train_dir, output_dir, walltimes_csv)
    else:
        walltimes_csv = Path(args.walltimes_csv).resolve()
        if not walltimes_csv.is_file():
            raise FileNotFoundError('walltimes CSV does not exist: %s' % walltimes_csv)

    try:
        class_by_instance, train_classes, valid_classes = split_record_maps(
            train_dir, split_file)
        rows, unknown_instances = read_walltime_rows(walltimes_csv, class_by_instance)
        stats = class_stats(
            rows,
            timeout_ms=float(args.command_timeout_sec) * 1000.0,
            par_multiplier=args.par)
        class_policy, summary = build_policy(stats, train_classes, valid_classes, args)

        payload = {
            'schema_version': 1,
            'train_dir': str(train_dir),
            'split': 'valid',
            'decision_level': 'problem_class',
            'decision_metric': 'walltime_PAR%s' % ('%g' % args.par),
            'timeout_ms': float(args.command_timeout_sec) * 1000.0,
            'par_multiplier': args.par,
            'min_improvement': args.min_improvement,
            'allow_fewer_solved': args.allow_fewer_solved,
            'min_validation_instances': args.min_validation_instances,
            'small_class_decision': args.small_class_decision,
            'default_decision_for_unseen_class': args.unseen_class_decision,
            'unknown_instances_in_walltime_csv': unknown_instances,
            'summary': summary,
            'classes': class_policy,
        }
        write_policy(policy_output, payload)

        model_classes = sum(1 for row in class_policy.values()
                            if row['decision'] == 'model')
        default_classes = sum(1 for row in class_policy.values()
                              if row['decision'] == 'default')
        print('Policy classes: model=%d default=%d' %
              (model_classes, default_classes))
    finally:
        if temp_context is not None:
            temp_context.cleanup()


if __name__ == '__main__':
    main()
