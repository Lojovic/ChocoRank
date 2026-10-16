from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import contextlib
import json
import os
import shlex
import subprocess
import tempfile

os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'rank-matplotlib'))
if not os.path.isdir(os.environ['MPLCONFIGDIR']):
    os.makedirs(os.environ['MPLCONFIGDIR'])


@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stderr = os.dup(2)
        try:
            os.dup2(devnull.fileno(), 2)
            with contextlib.redirect_stderr(devnull):
                yield
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Predict Choco propagator priorities with NumPy weights and optionally run Choco.')
    parser.add_argument('--train_dir', default='tmp/csp_rank')
    parser.add_argument('--instance', default=None)
    parser.add_argument('--instance_file', default=None)
    parser.add_argument('--input_json', default=None)
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--instance_features', default=None)
    parser.add_argument('--feature_extractor', default=None)
    parser.add_argument('--feature_output', default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--prediction_output', default=None)
    parser.add_argument('--priorities_output', default=None)
    parser.add_argument('--weights', default=None,
                        help='Optional NumPy weights path. Defaults to <train_dir>/numpy_weights.npz.')
    parser.add_argument('--choco_cmd', default=None,
                        help='Command template. Available placeholders: '
                             '{instance}, {prediction_file}, {priorities_file}')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def make_predict_args(args):
    predict_args = argparse.Namespace()
    predict_args.train_dir = args.train_dir
    predict_args.instance = args.instance
    predict_args.instance_file = args.instance_file
    predict_args.input_json = args.input_json
    predict_args.dataset = args.dataset
    predict_args.instance_features = args.instance_features
    predict_args.feature_extractor = args.feature_extractor
    predict_args.feature_output = args.feature_output
    predict_args.weights = args.weights
    predict_args.output = None
    return predict_args


def default_output_paths(args, prediction):
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='csp_priorities_')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    safe_instance = prediction['instance'].replace('/', '_').replace('\\', '_')
    prediction_output = args.prediction_output
    if prediction_output is None:
        prediction_output = os.path.join(output_dir, safe_instance + '.prediction.json')
    priorities_output = args.priorities_output
    if priorities_output is None:
        priorities_output = os.path.join(output_dir, safe_instance + '.priorities.json')
    return prediction_output, priorities_output


def write_json(path, payload):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(path, 'w') as fout:
        json.dump(payload, fout, indent=2, sort_keys=True)
        fout.write('\n')


def priorities_payload(prediction):
    return {
        'instance': prediction['instance'],
        'priorities': prediction['priorities'],
        'priority_groups': prediction['priority_groups'],
    }


def format_choco_command(command_template, prediction, prediction_file, priorities_file):
    return command_template.format(
        instance=prediction['instance'],
        prediction_file=prediction_file,
        priorities_file=priorities_file)


def main():
    args = parse_args()
    from predict_priorities import predict
    if args.verbose:
        prediction = predict(make_predict_args(args))
    else:
        with suppress_stderr():
            prediction = predict(make_predict_args(args))
    prediction_output, priorities_output = default_output_paths(args, prediction)

    write_json(prediction_output, prediction)
    write_json(priorities_output, priorities_payload(prediction))

    if args.verbose:
        print('prediction_file=%s' % prediction_output)
        print('priorities_file=%s' % priorities_output)

    if args.choco_cmd is None:
        return

    command = format_choco_command(
        args.choco_cmd,
        prediction,
        prediction_output,
        priorities_output)
    if args.verbose or args.dry_run:
        print('choco_cmd=%s' % command)
    if args.dry_run:
        return

    completed = subprocess.call(shlex.split(command))
    if completed != 0:
        raise SystemExit(completed)


if __name__ == '__main__':
    main()
