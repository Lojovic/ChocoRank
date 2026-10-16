#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def parse_args():
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description='Extract xcsp2feat features for a directory of XCSP XML instances.')
    parser.add_argument('--instances-dir', required=True,
                        help='Directory searched recursively for .xml instances.')
    parser.add_argument('--output', default=str(project_root / 'data/instance_features.csv'),
                        help='Output CSV. Defaults to data/instance_features.csv.')
    parser.add_argument('--feature-extractor',
                        default=str(project_root / 'feature_extractor/xcsp2feat'))
    parser.add_argument('--limit', type=int, default=None,
                        help='Extract at most this many instances, after sorting by basename.')
    parser.add_argument('--progress-every', type=int, default=50)
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


def run_text(command):
    return subprocess.check_output(command, text=True).strip()


def main():
    args = parse_args()
    extractor = Path(args.feature_extractor).resolve()
    if not extractor.is_file():
        raise FileNotFoundError('feature extractor not found: %s' % extractor)

    instances = list_instances(args.instances_dir, args.limit)
    header = run_text([str(extractor), '-p'])
    fieldnames = next(csv.reader([header]))
    if not fieldnames or fieldnames[0] != 'instance_name':
        raise ValueError('unexpected feature header; first column should be instance_name')

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as fout:
        fout.write(header)
        fout.write('\n')
        for index, instance in enumerate(instances, start=1):
            row = run_text([str(extractor), '-i', str(instance)])
            values = next(csv.reader([row]))
            if len(values) != len(fieldnames):
                raise ValueError(
                    'feature row for %s has %d columns, expected %d' %
                    (instance, len(values), len(fieldnames)))
            fout.write(row)
            fout.write('\n')
            if args.progress_every and (index == 1 or index % args.progress_every == 0):
                print('[%d/%d] %s' % (index, len(instances), instance.name))

    print('Wrote %s rows to %s' % (len(instances), output))


if __name__ == '__main__':
    main()
