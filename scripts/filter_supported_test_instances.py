#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


def parse_args():
    bundle_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description='Probe XCSP instances and keep only those whose propagator '
                    'classes are supported by a trained Rank model.')
    parser.add_argument('--train-dir', default='models/all_instances',
                        help='Model train_dir containing metadata.json and split files.')
    parser.add_argument('--split', default='test',
                        help='Split name to probe/filter. Defaults to test.')
    parser.add_argument('--instances-dir', default=None,
                        help='Optional directory for resolving split instance names by basename.')
    parser.add_argument('--jar',
                        default=str(bundle_dir / 'choco/dist/choco-rank.jar'),
                        help='Choco Rank jar with --probe support.')
    parser.add_argument('--probe-timeout-sec', type=float, default=120.0,
                        help='Timeout for each Java --probe command.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Probe at most this many split rows.')
    parser.add_argument('--progress-every', type=int, default=25,
                        help='Print progress every N probes.')
    parser.add_argument('--report', default=None,
                        help='JSON report path. Defaults to <train-dir>/supported_propagator_probe_report.json.')
    parser.add_argument('--unsupported-csv', default=None,
                        help='CSV path for removed/failed instances. Defaults to <train-dir>/unsupported_test_instances.csv.')
    parser.add_argument('--update-split', action='store_true',
                        help='Rewrite split_instances.csv/json so the selected split contains only supported rows.')
    return parser.parse_args()


def problem_class(instance):
    name = Path(instance).name
    if name.endswith('.xml'):
        name = name[:-4]
    if '-' in name:
        return name.split('-', 1)[0]
    return name


def load_split_rows(csv_path):
    with open(csv_path, newline='') as fin:
        reader = csv.DictReader(fin)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise ValueError('No split rows found in %s' % csv_path)
    return rows, fieldnames


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


def run_probe(jar, instance_path, output_path, timeout_sec):
    command = [
        'java',
        '-jar',
        str(jar),
        '--probe',
        str(output_path),
        str(instance_path),
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_sec)
    return completed.returncode, completed.stdout


def probe_row(row, jar, supported, instance_index, instances_dir, probe_dir, timeout_sec):
    instance_value = row['instance']
    try:
        instance_path = resolve_instance(instance_value, instances_dir, instance_index)
    except FileNotFoundError as exc:
        return {
            'status': 'unresolved',
            'instance': instance_value,
            'problem_class': row.get('problem_class') or problem_class(instance_value),
            'unsupported_propagators': [],
            'propagators': [],
            'message': str(exc),
        }

    probe_path = probe_dir / (instance_path.name + '.probe.json')
    try:
        return_code, output = run_probe(jar, instance_path, probe_path, timeout_sec)
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout if isinstance(exc.stdout, str) else ''
        return {
            'status': 'probe_timeout',
            'instance': instance_value,
            'resolved_instance': str(instance_path),
            'problem_class': row.get('problem_class') or problem_class(instance_value),
            'unsupported_propagators': [],
            'propagators': [],
            'message': (output or '')[-4000:],
        }

    if return_code != 0:
        return {
            'status': 'probe_failed',
            'instance': instance_value,
            'resolved_instance': str(instance_path),
            'problem_class': row.get('problem_class') or problem_class(instance_value),
            'unsupported_propagators': [],
            'propagators': [],
            'message': output[-4000:],
        }

    if not probe_path.is_file():
        return {
            'status': 'probe_missing_output',
            'instance': instance_value,
            'resolved_instance': str(instance_path),
            'problem_class': row.get('problem_class') or problem_class(instance_value),
            'unsupported_propagators': [],
            'propagators': [],
            'message': output[-4000:],
        }

    with open(probe_path) as fin:
        payload = json.load(fin)
    propagators = sorted(payload.get('propagators', []))
    unsupported = sorted(set(propagators) - supported)
    status = 'supported' if not unsupported else 'unsupported'
    return {
        'status': status,
        'instance': instance_value,
        'resolved_instance': str(instance_path),
        'problem_class': row.get('problem_class') or problem_class(instance_value),
        'unsupported_propagators': unsupported,
        'propagators': propagators,
        'runtime_ms': payload.get('runtime_ms'),
        'message': output[-4000:],
    }


def write_split_files(csv_path, json_path, rows, fieldnames):
    with open(csv_path, 'w', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, 'w') as fout:
        json.dump(rows, fout, indent=2, sort_keys=True)
        fout.write('\n')


def write_removed_csv(path, removed):
    fieldnames = [
        'instance',
        'resolved_instance',
        'problem_class',
        'status',
        'unsupported_propagators',
        'message',
    ]
    with open(path, 'w', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for item in removed:
            writer.writerow({
                'instance': item.get('instance', ''),
                'resolved_instance': item.get('resolved_instance', ''),
                'problem_class': item.get('problem_class', ''),
                'status': item.get('status', ''),
                'unsupported_propagators': ','.join(item.get('unsupported_propagators', [])),
                'message': item.get('message', '').replace('\n', '\\n'),
            })


def main():
    args = parse_args()
    train_dir = Path(args.train_dir).resolve()
    csv_path = train_dir / 'split_instances.csv'
    json_path = train_dir / 'split_instances.json'
    metadata_path = train_dir / 'metadata.json'
    report_path = Path(args.report).resolve() if args.report else train_dir / 'supported_propagator_probe_report.json'
    unsupported_csv = (
        Path(args.unsupported_csv).resolve()
        if args.unsupported_csv else train_dir / 'unsupported_test_instances.csv')

    jar = Path(args.jar).resolve()
    if not jar.is_file():
        raise FileNotFoundError('Missing probe jar: %s' % jar)
    with open(metadata_path) as fin:
        supported = set(json.load(fin)['propagators'])
    rows, fieldnames = load_split_rows(csv_path)
    selected = [row for row in rows if row.get('split') == args.split]
    if args.limit is not None:
        selected = selected[:args.limit]

    instance_index = build_instance_index(args.instances_dir)
    kept_instances = set()
    removed = []
    supported_count = 0
    status_counts = collections.Counter()
    unsupported_counts = collections.Counter()
    start = time.time()

    with tempfile.TemporaryDirectory(prefix='rank-probes-') as tmp:
        probe_dir = Path(tmp)
        for index, row in enumerate(selected, start=1):
            result = probe_row(
                row,
                jar,
                supported,
                instance_index,
                args.instances_dir,
                probe_dir,
                args.probe_timeout_sec)
            status_counts[result['status']] += 1
            if result['status'] == 'supported':
                kept_instances.add(row['instance'])
                supported_count += 1
            else:
                removed.append(result)
                unsupported_counts.update(result.get('unsupported_propagators', []))

            if index == 1 or index % args.progress_every == 0 or index == len(selected):
                elapsed = time.time() - start
                print(
                    'probed %d/%d supported=%d removed=%d elapsed=%.1fs' %
                    (index, len(selected), supported_count, len(removed), elapsed),
                    flush=True)

    report = {
        'train_dir': str(train_dir),
        'split': args.split,
        'selected_count': len(selected),
        'supported_count': supported_count,
        'removed_count': len(removed),
        'status_counts': dict(status_counts),
        'unsupported_propagator_counts': dict(unsupported_counts.most_common()),
        'removed': removed,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as fout:
        json.dump(report, fout, indent=2, sort_keys=True)
        fout.write('\n')
    write_removed_csv(unsupported_csv, removed)

    if args.update_split:
        new_rows = []
        for row in rows:
            if row.get('split') != args.split or row['instance'] in kept_instances:
                new_rows.append(row)
        write_split_files(csv_path, json_path, new_rows, fieldnames)
        print('Updated split files: %s and %s' % (csv_path, json_path))
    else:
        print('Dry run only. Add --update-split to rewrite split_instances.csv/json.')

    print('Report: %s' % report_path)
    print('Removed CSV: %s' % unsupported_csv)


if __name__ == '__main__':
    main()
