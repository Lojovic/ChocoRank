from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('AUTOGRAPH_VERBOSITY', '0')
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'rank-matplotlib'))

@contextmanager
def suppress_stderr():
    saved_stderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        os.close(devnull)

import data_utils

import numpy as np
with suppress_stderr():
    import tensorflow as raw_tf
    raw_tf.get_logger().setLevel(logging.ERROR)
    raw_tf.compat.v1.logging.set_verbosity(raw_tf.compat.v1.logging.ERROR)
    raw_tf.compat.v1.disable_v2_behavior()
    from rank_model import CspRank
tf = raw_tf.compat.v1


def parse_args():
    parser = argparse.ArgumentParser(description='Train CSP Rank.')
    parser.add_argument('--dataset', default='data/dataset.json')
    parser.add_argument('--instance_features', default='data/instance_features.csv')
    parser.add_argument('--train_dir', default='tmp/csp_rank')
    parser.add_argument('--hparams', default='')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=100)
    parser.add_argument('--valid_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--min_pair_runtime_ratio', type=float, default=0.0)
    parser.add_argument('--pair_sampling_strategy',
                        choices=[
                            'weighted',
                            'threshold',
                            'all_pairs',
                            'topk_bottomk',
                        ],
                        default='weighted')
    parser.add_argument('--pair_keep_floor', type=float, default=0.1)
    parser.add_argument('--pair_sampling_seed', type=int, default=0)
    parser.add_argument('--pair_top_k', type=int, default=3)
    parser.add_argument('--split_strategy',
                        choices=[
                            'random',
                            'within_class_stratified',
                            'within_class_bucket_stratified',
                        ],
                        default='within_class_stratified')
    parser.add_argument('--difficulty_buckets', type=int, default=3)
    parser.add_argument('--timeout_ms', type=float, default=20 * 60 * 1000)
    parser.add_argument('--timeout_penalty_multiplier', type=float, default=1.0)
    parser.add_argument('--validation_timeout_penalty_multiplier', type=float, default=1.0)
    parser.add_argument('--min_class_instances', type=int, default=7)
    parser.add_argument('--small_class_policy',
                        choices=['train', 'test', 'exclude'],
                        default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tie_seed', type=int, default=0)
    return parser.parse_args()


def split_examples(examples, valid_ratio, test_ratio, seed):
    if valid_ratio < 0 or test_ratio < 0 or valid_ratio + test_ratio >= 1:
        raise ValueError('valid_ratio and test_ratio must be non-negative and sum to < 1')

    shuffled = list(examples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)
    valid_count = int(round(total * valid_ratio))
    test_count = int(round(total * test_ratio))
    train_count = total - valid_count - test_count
    if train_count <= 0:
        raise ValueError('Split leaves no training examples')

    train = shuffled[:train_count]
    valid = shuffled[train_count:train_count + valid_count]
    test = shuffled[train_count + valid_count:]
    return train, valid, test


def problem_class(instance):
    name = os.path.basename(instance)
    if name.endswith('.xml'):
        name = name[:-4]
    if '-' in name:
        return name.split('-', 1)[0]
    return name


def median(values):
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return 0.0
    middle = count // 2
    if count % 2:
        return float(ordered[middle])
    return float(ordered[middle - 1] + ordered[middle]) / 2.0


def difficulty_key(example, timeout_ms):
    runtimes = [float(run.get('raw_runtime_ms', run['runtime_ms']))
                for run in example['runs']]
    timeout_count = sum(1 for runtime in runtimes if runtime >= timeout_ms)
    timeout_rate = float(timeout_count) / float(len(runtimes)) if runtimes else 0.0
    return timeout_rate, median(runtimes)


def effective_runtime_ms(runtime_ms, timeout_ms, timeout_penalty_multiplier):
    if runtime_ms >= timeout_ms:
        return timeout_penalty_multiplier * timeout_ms
    return runtime_ms


def split_counts(total, valid_ratio, test_ratio):
    if total <= 0:
        return 0, 0, 0
    valid_count = int(round(total * valid_ratio))
    test_count = int(round(total * test_ratio))
    if total >= 3:
        if valid_ratio > 0 and valid_count == 0:
            valid_count = 1
        if test_ratio > 0 and test_count == 0:
            test_count = 1
    if valid_count + test_count >= total:
        overflow = valid_count + test_count - (total - 1)
        while overflow > 0 and (valid_count > test_count or test_count == 0):
            valid_count -= 1
            overflow -= 1
        while overflow > 0 and test_count > 0:
            test_count -= 1
            overflow -= 1
    train_count = total - valid_count - test_count
    return train_count, valid_count, test_count


def hard_test_train_valid_split(class_examples, valid_ratio, test_ratio, rng):
    ordered = sorted(
        class_examples,
        key=lambda item: (
            item['difficulty_timeout_rate'],
            item['difficulty_median_runtime_ms'],
            item['example']['instance']))
    total = len(ordered)
    train_count, valid_count, test_count = split_counts(
        total, valid_ratio, test_ratio)

    train = ordered[:train_count]
    valid = ordered[train_count:train_count + valid_count]
    test = ordered[train_count + valid_count:train_count + valid_count + test_count]

    if len(valid) != valid_count or len(train) != train_count:
        raise ValueError('Internal split error: expected train/valid/test '
                         '%d/%d/%d, got %d/%d/%d' %
                         (train_count, valid_count, test_count,
                          len(train), len(valid), len(test)))
    return train, valid, test


def stratified_bucket_split(bucket_examples, valid_ratio, test_ratio, rng):
    shuffled = list(bucket_examples)
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_count, valid_count, test_count = split_counts(
        total, valid_ratio, test_ratio)
    train = shuffled[:train_count]
    valid = shuffled[train_count:train_count + valid_count]
    test = shuffled[train_count + valid_count:train_count + valid_count + test_count]
    return train, valid, test


def difficulty_buckets_for_class(class_examples, bucket_count):
    ordered = sorted(
        class_examples,
        key=lambda item: (
            item['difficulty_timeout_rate'],
            item['difficulty_median_runtime_ms'],
            item['example']['instance']))
    bucket_count = max(1, min(bucket_count, len(ordered)))
    buckets = [[] for _ in range(bucket_count)]
    for index, item in enumerate(ordered):
        bucket_index = int(math.floor(float(index) * bucket_count / len(ordered)))
        if bucket_index >= bucket_count:
            bucket_index = bucket_count - 1
        buckets[bucket_index].append(item)
    return buckets


def difficulty_bucket_name(bucket_index, bucket_count):
    if bucket_count == 3:
        return ['easy', 'medium', 'hard'][bucket_index]
    return 'bucket_%d' % bucket_index


def split_examples_within_class_stratified(
        examples,
        valid_ratio,
        test_ratio,
        seed,
        difficulty_bucket_count,
        timeout_ms,
        min_class_instances,
        small_class_policy,
        split_strategy):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for example in examples:
        timeout_rate, median_runtime = difficulty_key(example, timeout_ms)
        cls = problem_class(example['instance'])
        by_class[cls].append({
            'example': example,
            'problem_class': cls,
            'difficulty_timeout_rate': timeout_rate,
            'difficulty_median_runtime_ms': median_runtime,
        })

    train = []
    valid = []
    test = []
    split_records = []
    class_summaries = {}

    for cls in sorted(by_class):
        class_items = by_class[cls]
        class_summaries[cls] = {
            'instance_count': len(class_items),
            'policy': 'stratified',
        }
        if len(class_items) < min_class_instances:
            class_summaries[cls]['policy'] = small_class_policy
            if small_class_policy == 'train':
                train.extend(item['example'] for item in class_items)
                class_summaries[cls]['train_count'] = len(class_items)
                class_summaries[cls]['valid_count'] = 0
                class_summaries[cls]['test_count'] = 0
                class_summaries[cls]['excluded_count'] = 0
                for item in class_items:
                    split_records.append((item, 'train', 'small_class'))
            elif small_class_policy == 'test':
                test.extend(item['example'] for item in class_items)
                class_summaries[cls]['train_count'] = 0
                class_summaries[cls]['valid_count'] = 0
                class_summaries[cls]['test_count'] = len(class_items)
                class_summaries[cls]['excluded_count'] = 0
                for item in class_items:
                    split_records.append((item, 'test', 'small_class'))
            elif small_class_policy == 'exclude':
                class_summaries[cls]['train_count'] = 0
                class_summaries[cls]['valid_count'] = 0
                class_summaries[cls]['test_count'] = 0
                class_summaries[cls]['excluded_count'] = len(class_items)
                for item in class_items:
                    split_records.append((item, 'excluded', 'small_class'))
            else:
                raise ValueError('Unsupported small_class_policy: %s' %
                                 small_class_policy)
            continue

        if split_strategy == 'within_class_bucket_stratified':
            class_summaries[cls]['policy'] = 'bucket_stratified'
            class_train = []
            class_valid = []
            class_test = []
            buckets = difficulty_buckets_for_class(
                class_items, difficulty_bucket_count)
            for bucket_index, bucket_items in enumerate(buckets):
                bucket_name = difficulty_bucket_name(
                    bucket_index, len(buckets))
                bucket_train, bucket_valid, bucket_test = stratified_bucket_split(
                    bucket_items, valid_ratio, test_ratio, rng)
                class_train.extend(bucket_train)
                class_valid.extend(bucket_valid)
                class_test.extend(bucket_test)
                for item in bucket_train:
                    split_records.append((item, 'train', bucket_name))
                for item in bucket_valid:
                    split_records.append((item, 'valid', bucket_name))
                for item in bucket_test:
                    split_records.append((item, 'test', bucket_name))
        else:
            class_train, class_valid, class_test = hard_test_train_valid_split(
                class_items, valid_ratio, test_ratio, rng)
            for item in class_train:
                split_records.append((item, 'train', 'easy'))
            for item in class_valid:
                split_records.append((item, 'valid', 'medium'))
            for item in class_test:
                split_records.append((item, 'test', 'hard'))

        train.extend(item['example'] for item in class_train)
        valid.extend(item['example'] for item in class_valid)
        test.extend(item['example'] for item in class_test)
        class_train_count = len(class_train)
        class_valid_count = len(class_valid)
        class_test_count = len(class_test)
        class_summaries[cls]['train_count'] = class_train_count
        class_summaries[cls]['valid_count'] = class_valid_count
        class_summaries[cls]['test_count'] = class_test_count
        class_summaries[cls]['excluded_count'] = 0

    if not train:
        raise ValueError('Split leaves no training examples')
    return train, valid, test, split_records, class_summaries


def write_split_files(train_dir, split_records):
    json_path = os.path.join(train_dir, 'split_instances.json')
    csv_path = os.path.join(train_dir, 'split_instances.csv')
    records = []
    for item, split, bucket in sorted(
            split_records,
            key=lambda row: (row[1], row[0]['problem_class'], row[0]['example']['instance'])):
        records.append({
            'instance': item['example']['instance'],
            'problem_class': item['problem_class'],
            'split': split,
            'difficulty_bucket': bucket,
            'difficulty_timeout_rate': item['difficulty_timeout_rate'],
            'difficulty_median_runtime_ms': item['difficulty_median_runtime_ms'],
            'run_count': len(item['example']['runs']),
            'pair_count': len(item['example']['pairs']),
        })

    with open(json_path, 'w') as fout:
        json.dump(records, fout, indent=2, sort_keys=True)

    with open(csv_path, 'w', newline='') as fout:
        fieldnames = [
            'instance',
            'problem_class',
            'split',
            'difficulty_bucket',
            'difficulty_timeout_rate',
            'difficulty_median_runtime_ms',
            'run_count',
            'pair_count',
        ]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def should_export_variable(name):
    if name in ('Variable', 'Variable_1', 'beta1_power', 'beta2_power'):
        return False
    if name.endswith('/Adam') or name.endswith('/Adam_1'):
        return False
    if '/seed_generator' in name:
        return False
    return True


def export_numpy_weights(session, output_path):
    variables = {}
    for variable in tf.global_variables():
        name = variable.name.split(':', 1)[0]
        if should_export_variable(name):
            variables[name] = session.run(variable).astype(np.float32)
    np.savez(output_path, **variables)
    print('Exported %d tensors to %s' % (len(variables), output_path))
    sys.stdout.flush()


class ShuffledBatchIterator(object):
    def __init__(self, examples, batch_size, rng):
        self.examples = list(examples)
        self.batch_size = batch_size
        self.rng = rng
        self.position = 0
        self.rng.shuffle(self.examples)

    def next_batch(self):
        if len(self.examples) <= self.batch_size:
            return list(self.examples)

        batch = []
        while len(batch) < self.batch_size:
            remaining = len(self.examples) - self.position
            take = min(self.batch_size - len(batch), remaining)
            batch.extend(self.examples[self.position:self.position + take])
            self.position += take
            if self.position >= len(self.examples):
                self.rng.shuffle(self.examples)
                self.position = 0
        return batch


def iter_batches(examples, batch_size):
    for start in range(0, len(examples), batch_size):
        yield examples[start:start + batch_size]


def evaluate(session, model, examples, normalizer, args, max_prop_count,
             max_run_count, max_pair_count):
    if not examples:
        return None

    total_loss = 0.0
    total_weight = 0.0
    observed_expected_runtime_sum = 0.0
    for batch_examples in iter_batches(examples, args.batch_size):
        batch = data_utils.make_csp_training_batch(
            batch_examples,
            pad_id=model.pad_id,
            max_prop_count=max_prop_count,
            max_run_count=max_run_count,
            max_pair_count=max_pair_count,
            feature_normalizer=normalizer)
        loss, run_log_probs, _ = model.evaluate_batch(session, batch)
        batch_weight = float(np.sum(batch['pair_mask']))
        total_loss += float(loss) * batch_weight
        total_weight += batch_weight
        for row_index, example in enumerate(batch_examples):
            run_count = len(example['runs'])
            observed_log_probs = run_log_probs[row_index, :run_count]
            stable_log_probs = observed_log_probs - np.max(observed_log_probs)
            observed_probs = np.exp(stable_log_probs)
            observed_probs /= max(float(np.sum(observed_probs)), 1e-12)
            runtimes = np.array(
                [effective_runtime_ms(
                    float(run.get('raw_runtime_ms', run['runtime_ms'])),
                    args.timeout_ms,
                    args.validation_timeout_penalty_multiplier)
                 for run in example['runs']],
                dtype=np.float32)
            expected_runtime = float(np.sum(observed_probs * runtimes))
            observed_expected_runtime_sum += expected_runtime

    avg_loss = total_loss / total_weight if total_weight > 0 else 0.0
    return avg_loss, observed_expected_runtime_sum


def write_metadata(path, data, normalizer, args, split_sizes,
                   max_run_count, max_pair_count, class_split_summary):
    metadata = {
        'dataset': args.dataset,
        'instance_features': args.instance_features,
        'hparams': args.hparams,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'tie_seed': args.tie_seed,
        'split_strategy': args.split_strategy,
        'valid_ratio': args.valid_ratio,
        'test_ratio': args.test_ratio,
        'difficulty_buckets': args.difficulty_buckets,
        'difficulty_metric': {
            'primary': 'timeout_rate',
            'secondary': 'median_runtime_ms',
            'timeout_ms': args.timeout_ms,
        },
        'timeout_penalty_multiplier': args.timeout_penalty_multiplier,
        'validation_timeout_penalty_multiplier': args.validation_timeout_penalty_multiplier,
        'pair_weighting': 'log1p_runtime_delta_ratio',
        'min_class_instances': args.min_class_instances,
        'small_class_policy': args.small_class_policy,
        'class_split_summary': class_split_summary,
        'min_pair_runtime_ratio': args.min_pair_runtime_ratio,
        'pair_sampling_strategy': args.pair_sampling_strategy,
        'pair_keep_floor': args.pair_keep_floor,
        'pair_sampling_seed': args.pair_sampling_seed,
        'pair_top_k': args.pair_top_k,
        'split_sizes': split_sizes,
        'propagators': data.propagators,
        'instance_feature_names': data.instance_feature_names,
        'instance_feature_mean': normalizer.mean.tolist(),
        'instance_feature_std': normalizer.std.tolist(),
        'rank_list_size': data.rank_list_size,
        'max_run_count': max_run_count,
        'max_pair_count': max_pair_count,
    }
    with open(path, 'w') as fout:
        json.dump(metadata, fout, indent=2, sort_keys=True)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    print('Reading CSP data...')
    data = data_utils.read_csp_runtime_data(
        args.dataset,
        tie_seed=args.tie_seed,
        instance_feature_path=args.instance_features,
        min_pair_runtime_ratio=args.min_pair_runtime_ratio,
        pair_sampling_strategy=args.pair_sampling_strategy,
        pair_keep_floor=args.pair_keep_floor,
        pair_sampling_seed=args.pair_sampling_seed,
        pair_top_k=args.pair_top_k,
        timeout_ms=args.timeout_ms,
        timeout_penalty_multiplier=args.timeout_penalty_multiplier)
    split_records = []
    class_split_summary = {}
    if args.split_strategy == 'random':
        train_examples, valid_examples, test_examples = split_examples(
            data.examples, args.valid_ratio, args.test_ratio, args.seed)
        for split_name, split_examples_for_name in [
                ('train', train_examples),
                ('valid', valid_examples),
                ('test', test_examples)]:
            for example in split_examples_for_name:
                timeout_rate, median_runtime = difficulty_key(
                    example, args.timeout_ms)
                split_records.append(({
                    'example': example,
                    'problem_class': problem_class(example['instance']),
                    'difficulty_timeout_rate': timeout_rate,
                    'difficulty_median_runtime_ms': median_runtime,
                }, split_name, 'random'))
    else:
        train_examples, valid_examples, test_examples, split_records, class_split_summary = (
            split_examples_within_class_stratified(
                data.examples,
                args.valid_ratio,
                args.test_ratio,
                args.seed,
                args.difficulty_buckets,
                args.timeout_ms,
                args.min_class_instances,
                args.small_class_policy,
                args.split_strategy))
    normalizer = data_utils.fit_csp_feature_normalizer(train_examples)

    max_run_count = max(len(example['runs']) for example in data.examples)
    max_pair_count = max(len(example['pairs']) for example in data.examples)
    split_sizes = {
        'train': len(train_examples),
        'valid': len(valid_examples),
        'test': len(test_examples),
    }

    print('Examples:', split_sizes)
    print('Propagators:', len(data.propagators))
    print('Instance features:', data.instance_feature_size)
    print('Max propagators per instance:', data.rank_list_size)
    print('Max runs per instance:', max_run_count)
    print('Max pairs per instance:', max_pair_count)
    print('Pair runtime ratio reference:', args.min_pair_runtime_ratio)
    print('Pair sampling strategy:', args.pair_sampling_strategy)
    print('Pair keep floor:', args.pair_keep_floor)
    print('Pair top K:', args.pair_top_k)
    print('Training timeout penalty multiplier:', args.timeout_penalty_multiplier)
    print('Validation timeout penalty multiplier:', args.validation_timeout_penalty_multiplier)
    print('Total kept pairs:', data.pair_count)

    if not os.path.isdir(args.train_dir):
        os.makedirs(args.train_dir)
    write_split_files(args.train_dir, split_records)

    tf.reset_default_graph()
    model = CspRank(
        num_propagators=len(data.propagators),
        max_prop_count=data.rank_list_size,
        batch_size=args.batch_size,
        hparam_str=args.hparams,
        max_run_count=max_run_count,
        max_pair_count=max_pair_count,
        instance_feature_size=data.instance_feature_size)

    rng = random.Random(args.seed)
    train_batch_iterator = ShuffledBatchIterator(
        train_examples, args.batch_size, rng)
    best_valid_expected_runtime = None
    checkpoint_prefix = os.path.join(args.train_dir, 'CspRank.ckpt')
    metadata_path = os.path.join(args.train_dir, 'metadata.json')
    numpy_weights_path = os.path.join(args.train_dir, 'numpy_weights.npz')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with suppress_stderr():
        session = tf.Session(config=config)
        session.run(tf.global_variables_initializer())
    with session:
        write_metadata(
            metadata_path,
            data,
            normalizer,
            args,
            split_sizes,
            max_run_count,
            max_pair_count,
            class_split_summary)

        start_time = time.time()
        for step in range(1, args.max_steps + 1):
            batch_examples = train_batch_iterator.next_batch()
            batch = data_utils.make_csp_training_batch(
                batch_examples,
                pad_id=model.pad_id,
                max_prop_count=data.rank_list_size,
                max_run_count=max_run_count,
                max_pair_count=max_pair_count,
                feature_normalizer=normalizer)
            grad_norm, train_loss = model.train_step(session, batch)

            if step == 1 or step % args.eval_every == 0 or step == args.max_steps:
                valid_metrics = evaluate(
                    session,
                    model,
                    valid_examples,
                    normalizer,
                    args,
                    data.rank_list_size,
                    max_run_count,
                    max_pair_count)
                elapsed = time.time() - start_time
                if valid_metrics is None:
                    print('step %d train_loss %.6f grad_norm %.6f elapsed %.1fs' %
                          (step, train_loss, grad_norm, elapsed))
                    if step == args.max_steps:
                        model.saver.save(session, checkpoint_prefix, global_step=step)
                        export_numpy_weights(session, numpy_weights_path)
                    continue

                valid_loss, valid_expected_runtime = valid_metrics
                print(
                    'step %d train_loss %.6f grad_norm %.6f '
                    'valid_loss %.6f valid_expected_runtime %.6f elapsed %.1fs' %
                    (step, train_loss, grad_norm, valid_loss,
                     valid_expected_runtime, elapsed))
                sys.stdout.flush()

                if (best_valid_expected_runtime is None or
                        valid_expected_runtime < best_valid_expected_runtime):
                    best_valid_expected_runtime = valid_expected_runtime
                    model.saver.save(session, checkpoint_prefix, global_step=step)
                    export_numpy_weights(session, numpy_weights_path)

        test_metrics = evaluate(
            session,
            model,
            test_examples,
            normalizer,
            args,
            data.rank_list_size,
            max_run_count,
            max_pair_count)
        if test_metrics is not None:
            test_loss, test_expected_runtime = test_metrics
            print('test_loss %.6f test_expected_runtime %.6f' %
                  (test_loss, test_expected_runtime))


if __name__ == '__main__':
    main()
