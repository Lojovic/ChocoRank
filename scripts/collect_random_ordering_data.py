#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path


CHOCO_PRIORITIES = [
    'UNARY',
    'BINARY',
    'TERNARY',
    'LINEAR',
    'QUADRATIC',
    'CUBIC',
    'VERY_SLOW',
]


def parse_args():
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description='Collect runtime data by solving each XCSP instance with random propagator priority orderings.')
    parser.add_argument('--instances-dir', required=True,
                        help='Directory searched recursively for .xml instances.')
    parser.add_argument('--output', default=str(project_root / 'data/dataset.json'),
                        help='Output dataset JSON. Defaults to data/dataset.json.')
    parser.add_argument('--jar', default=str(project_root / 'choco/dist/choco-rank.jar'))
    parser.add_argument('--orderings', type=int, default=20,
                        help='Number of random priority orderings per instance.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timeout-sec', type=float, default=20 * 60,
                        help='Per solve timeout. Timed-out runs are recorded with runtime_ms=timeout.')
    parser.add_argument('--probe-timeout-sec', type=float, default=120.0)
    parser.add_argument('--limit', type=int, default=None,
                        help='Collect data for at most this many instances, after sorting by basename.')
    parser.add_argument('--append', action='store_true',
                        help='Append to an existing dataset JSON instead of replacing it.')
    parser.add_argument('--work-dir', default=None,
                        help='Directory for temporary probe/priority/result/log files. Defaults to a temp dir.')
    parser.add_argument('--keep-work-dir', action='store_true')
    parser.add_argument('--progress-every', type=int, default=1)
    return parser.parse_args()


def list_instances(instances_dir, limit):
    root = Path(instances_dir).resolve()
    if not root.is_dir():
        raise ValueError('instances-dir does not exist: %s' % root)
    instances = sorted(
        (path for path in root.rglob('*.xml') if path.is_file()),
        key=lambda path: (path.name, str(path)))
    if limit is not None:
        instances = instances[:limit]
    return instances


def run_command(command, timeout_sec):
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec)
        return completed.returncode, completed.stdout, False
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ''
        if isinstance(output, bytes):
            output = output.decode('utf-8', errors='replace')
        return -1, output, True


def read_json(path):
    with open(path) as fin:
        return json.load(fin)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as fout:
        json.dump(payload, fout, indent=2, sort_keys=True)
        fout.write('\n')


def priority_for_position(position, count):
    if count <= len(CHOCO_PRIORITIES):
        return CHOCO_PRIORITIES[position]
    priority_index = int(position * len(CHOCO_PRIORITIES) / count)
    return CHOCO_PRIORITIES[min(priority_index, len(CHOCO_PRIORITIES) - 1)]


def random_priorities(propagators, rng):
    ordering = list(propagators)
    rng.shuffle(ordering)
    return {
        propagator: priority_for_position(index, len(ordering))
        for index, propagator in enumerate(ordering)
    }


def numeric_scores_from_priorities(priorities):
    priority_to_score = dict((name, float(index)) for index, name in enumerate(CHOCO_PRIORITIES))
    return {
        propagator: priority_to_score[priority]
        for propagator, priority in priorities.items()
    }


def probe_instance(jar, instance, work_dir, timeout_sec):
    probe_json = work_dir / (instance.stem + '.probe.json')
    command = [
        'java',
        '-jar',
        str(jar),
        '--probe',
        str(probe_json),
        str(instance),
    ]
    code, output, timed_out = run_command(command, timeout_sec)
    if timed_out:
        raise RuntimeError('probe timed out for %s' % instance)
    if code != 0:
        raise RuntimeError('probe failed for %s:\n%s' % (instance, output[-4000:]))
    payload = read_json(probe_json)
    propagators = sorted(payload.get('propagators', []))
    if not propagators:
        raise RuntimeError('probe found no propagators for %s' % instance)
    return propagators


def run_ordering(jar, instance, priorities, work_dir, timeout_sec, timeout_ms, run_index):
    safe_name = instance.stem.replace('/', '_')
    priorities_json = work_dir / ('%s.%03d.priorities.json' % (safe_name, run_index))
    result_json = work_dir / ('%s.%03d.result.json' % (safe_name, run_index))
    write_json(priorities_json, {
        'instance': instance.name,
        'priorities': priorities,
    })

    command = [
        'java',
        '-jar',
        str(jar),
        '--run',
        str(result_json),
        str(priorities_json),
        str(instance),
    ]
    code, output, timed_out = run_command(command, timeout_sec)
    if timed_out:
        return {
            'instance': instance.name,
            'runtime_ms': timeout_ms,
            'propagator_scores': numeric_scores_from_priorities(priorities),
        }, 'timeout'
    if code != 0:
        raise RuntimeError('solve failed for %s ordering %d:\n%s' %
                           (instance, run_index, output[-4000:]))
    rows = read_json(result_json)
    if not rows:
        raise RuntimeError('empty result JSON for %s ordering %d' % (instance, run_index))
    return rows[0], 'ok'


def main():
    args = parse_args()
    jar = Path(args.jar).resolve()
    if not jar.is_file():
        raise FileNotFoundError('jar not found: %s' % jar)

    instances = list_instances(args.instances_dir, args.limit)
    output = Path(args.output).resolve()
    existing = []
    if args.append and output.is_file():
        existing = read_json(output)
        if not isinstance(existing, list):
            raise ValueError('existing dataset is not a JSON list: %s' % output)

    timeout_ms = int(round(args.timeout_sec * 1000.0))
    rng = random.Random(args.seed)
    created_temp = args.work_dir is None
    work_dir = Path(args.work_dir).resolve() if args.work_dir else Path(tempfile.mkdtemp(prefix='chocorank_collect_'))
    work_dir.mkdir(parents=True, exist_ok=True)

    records = list(existing)
    ok_runs = 0
    timeout_runs = 0
    try:
        for instance_index, instance in enumerate(instances, start=1):
            propagators = probe_instance(jar, instance, work_dir, args.probe_timeout_sec)
            for run_index in range(args.orderings):
                priorities = random_priorities(propagators, rng)
                row, status = run_ordering(
                    jar,
                    instance,
                    priorities,
                    work_dir,
                    args.timeout_sec,
                    timeout_ms,
                    run_index)
                records.append(row)
                if status == 'timeout':
                    timeout_runs += 1
                else:
                    ok_runs += 1
            write_json(output, records)
            if args.progress_every and (
                    instance_index == 1 or instance_index % args.progress_every == 0):
                print('[%d/%d] %s propagators=%d runs=%d timeouts=%d' %
                      (instance_index, len(instances), instance.name,
                       len(propagators), ok_runs + timeout_runs, timeout_runs))
    finally:
        if created_temp and not args.keep_work_dir:
            for path in sorted(work_dir.glob('*')):
                path.unlink()
            work_dir.rmdir()

    print('Wrote %d total rows to %s (%d new ok, %d new timeout)' %
          (len(records), output, ok_runs, timeout_runs))


if __name__ == '__main__':
    main()
