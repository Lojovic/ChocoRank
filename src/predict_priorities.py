from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import json
import os
import subprocess

import numpy as np

import data_utils


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
    parser = argparse.ArgumentParser(
        description='Predict CSP propagator priorities with NumPy-only inference.')
    parser.add_argument('--train_dir', default='tmp/csp_rank')
    parser.add_argument('--weights', default=None)
    parser.add_argument('--instance', default=None)
    parser.add_argument('--instance_file', default=None)
    parser.add_argument('--input_json', default=None)
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--instance_features', default=None)
    parser.add_argument('--feature_extractor', default=None)
    parser.add_argument('--feature_output', default=None)
    parser.add_argument('--output', default=None)
    return parser.parse_args()


def load_metadata(train_dir):
    metadata_path = os.path.join(train_dir, 'metadata.json')
    with open(metadata_path) as fin:
        return json.load(fin)


def parse_hparams(hparam_str):
    hparams = {
        'prop_embedding_size': 64,
        'hidden_units': 128,
        'num_blocks': 2,
        'num_heads': 4,
        'dropout_rate': 0.1,
        'activation': 'relu',
        'empty_bound': 1e9,
    }
    if hparam_str:
        for item in hparam_str.split(','):
            if not item:
                continue
            key, value = item.split('=', 1)
            if key not in hparams:
                continue
            old_value = hparams[key]
            hparams[key] = type(old_value)(value)
    return hparams


def load_input_from_json(path):
    with open(path) as fin:
        payload = json.load(fin)
    if 'instance' not in payload:
        raise ValueError('Input JSON must contain instance')
    if 'propagators' not in payload:
        raise ValueError('Input JSON must contain propagators')
    return payload


def get_dataset_example(instance, dataset_path, instance_feature_path):
    data = data_utils.read_csp_runtime_data(
        dataset_path,
        tie_seed=0,
        instance_feature_path=instance_feature_path)
    if instance not in data.example_by_instance:
        raise ValueError('Instance %s was not found in %s' % (instance, dataset_path))
    return data.example_by_instance[instance]


def collect_instance_features(instance_file, extractor_path, expected_feature_names,
                              feature_output=None):
    if instance_file is None:
        raise ValueError('Cannot collect instance features without --instance_file')
    if extractor_path is None:
        raise ValueError('Cannot collect instance features without --feature_extractor')

    header = subprocess.check_output(
        [extractor_path, '-p'], universal_newlines=True).strip()
    row = subprocess.check_output(
        [extractor_path, '-i', instance_file], universal_newlines=True).strip()

    if feature_output is not None:
        parent = os.path.dirname(feature_output)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)
        with open(feature_output, 'w') as fout:
            fout.write(header)
            fout.write('\n')
            fout.write(row)
            fout.write('\n')

    reader = csv.DictReader([header, row])
    extracted = list(reader)
    if len(extracted) != 1:
        raise ValueError('Feature extractor did not emit exactly one feature row')
    feature_names = [name for name in reader.fieldnames if name != 'instance_name']
    if feature_names != expected_feature_names:
        raise ValueError(
            'Extracted feature columns do not match model metadata. '
            'First mismatch: %s' %
            _first_feature_mismatch(feature_names, expected_feature_names))
    return [float(extracted[0][name]) for name in expected_feature_names]


def _first_feature_mismatch(actual, expected):
    for index, (actual_name, expected_name) in enumerate(zip(actual, expected)):
        if actual_name != expected_name:
            return 'column %d actual=%s expected=%s' % (
                index, actual_name, expected_name)
    if len(actual) != len(expected):
        return 'actual_count=%d expected_count=%d' % (len(actual), len(expected))
    return 'none'


def features_for_instance(instance, args, metadata, feature_path):
    if args.feature_extractor is not None:
        return collect_instance_features(
            args.instance_file,
            args.feature_extractor,
            metadata['instance_feature_names'],
            feature_output=args.feature_output)

    feature_map, feature_names = data_utils.read_instance_features_csv(feature_path)
    if instance not in feature_map:
        raise ValueError('Missing instance features for %s' % instance)
    if feature_names != metadata['instance_feature_names']:
        raise ValueError('Instance feature columns do not match metadata')
    return feature_map[instance]


def build_prediction_example(args, metadata):
    dataset_path = args.dataset or metadata['dataset']
    feature_path = args.instance_features or metadata['instance_features']
    prop_to_id = dict((name, index) for index, name in enumerate(metadata['propagators']))

    if args.input_json is not None:
        payload = load_input_from_json(args.input_json)
        instance = payload['instance']
        propagators = payload['propagators']
        if 'instance_features' in payload:
            instance_features = payload['instance_features']
        else:
            instance_features = features_for_instance(
                instance, args, metadata, feature_path)
        return data_utils.make_csp_prediction_example(
            instance, propagators, prop_to_id, instance_features)

    if args.instance is None:
        raise ValueError('Provide either --instance or --input_json')

    lookup_feature_path = None if args.feature_extractor is not None else feature_path
    example = get_dataset_example(args.instance, dataset_path, lookup_feature_path)
    instance_features = example['instance_features']
    if args.feature_extractor is not None:
        instance_features = features_for_instance(
            example['instance'], args, metadata, feature_path)
    return data_utils.make_csp_prediction_example(
        example['instance'],
        example['propagators'],
        prop_to_id,
        instance_features)


def group_scores_into_priorities(score_items):
    sorted_items = sorted(score_items, key=lambda item: (item[1], item[0]))
    groups = [[item] for item in sorted_items]

    while len(groups) > len(CHOCO_PRIORITIES):
        best_index = None
        best_gap = None
        for index in range(len(groups) - 1):
            left_score = groups[index][-1][1]
            right_score = groups[index + 1][0][1]
            gap = right_score - left_score
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_index = index
        groups[best_index].extend(groups[best_index + 1])
        del groups[best_index + 1]

    priorities = {}
    for group_index, group in enumerate(groups):
        priority = CHOCO_PRIORITIES[group_index]
        for propagator, _ in group:
            priorities[propagator] = priority
    return groups, priorities


def activation(values, name):
    if name == 'elu':
        return np.where(values > 0.0, values, np.expm1(values))
    return np.maximum(values, 0.0)


def dense(inputs, weights, prefix, activation_name=None):
    outputs = np.matmul(inputs, weights[prefix + '/kernel']) + weights[prefix + '/bias']
    if activation_name is not None:
        outputs = activation(outputs, activation_name)
    return outputs


def layer_norm(inputs, weights, prefix):
    mean = np.mean(inputs, axis=-1, keepdims=True)
    variance = np.mean(np.square(inputs - mean), axis=-1, keepdims=True)
    normalized = (inputs - mean) / np.sqrt(variance + 1e-8)
    return normalized * weights[prefix + '/gamma'] + weights[prefix + '/beta']


def multihead_attention(enc, mask, weights, prefix, hparams):
    hidden_units = hparams['hidden_units']
    num_heads = hparams['num_heads']
    head_size = hidden_units // num_heads
    activation_name = hparams['activation']

    q = dense(enc, weights, prefix + '/q', activation_name)
    k = dense(enc, weights, prefix + '/k', activation_name)
    v = dense(enc, weights, prefix + '/v', activation_name)

    q_heads = np.concatenate(np.split(q, num_heads, axis=2), axis=0)
    k_heads = np.concatenate(np.split(k, num_heads, axis=2), axis=0)
    v_heads = np.concatenate(np.split(v, num_heads, axis=2), axis=0)
    mask_heads = np.tile(mask, (num_heads, 1))

    logits = np.matmul(q_heads, np.swapaxes(k_heads, 1, 2)) / np.sqrt(float(head_size))
    key_mask = np.expand_dims(mask_heads, axis=1)
    logits = np.where(key_mask == 0.0, np.float32(-2 ** 32 + 1), logits)
    attn = softmax(logits, axis=2)
    query_mask = np.expand_dims(mask_heads, axis=2)
    attn = attn * query_mask

    outputs = np.matmul(attn, v_heads)
    outputs = np.concatenate(np.split(outputs, num_heads, axis=0), axis=2)
    return layer_norm(outputs + enc, weights, prefix + '/ln')


def feedforward(enc, weights, prefix, hparams):
    hidden = dense(enc, weights, prefix + '/inner', hparams['activation'])
    outputs = dense(hidden, weights, prefix + '/readout')
    return layer_norm(outputs + enc, weights, prefix + '/ln')


def softmax(values, axis):
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def score_batch(weights, batch, hparams):
    prop_ids = batch['propagator_ids'].astype(np.int64)
    mask = batch['propagator_mask'].astype(np.float32)
    instance_features = batch['instance_features'].astype(np.float32)

    prop_embeddings = weights['propagator_embeddings']
    pad_embedding = np.zeros((1, prop_embeddings.shape[1]), dtype=np.float32)
    prop_embeddings_with_pad = np.concatenate([prop_embeddings, pad_embedding], axis=0)

    enc = prop_embeddings_with_pad[prop_ids]
    enc = dense(enc, weights, 'input_projection', hparams['activation'])
    if instance_features.shape[1] > 0:
        instance_context = dense(
            instance_features, weights, 'instance_projection', hparams['activation'])
        instance_context = np.expand_dims(instance_context, axis=1)
        instance_context = np.tile(instance_context, (1, prop_ids.shape[1], 1))
        enc = dense(
            np.concatenate([enc, instance_context], axis=2),
            weights,
            'instance_conditioned_projection',
            hparams['activation'])

    for block_index in range(hparams['num_blocks']):
        block_prefix = 'csp_rank_block_%d' % block_index
        enc = multihead_attention(
            enc,
            mask,
            weights,
            block_prefix + '/self_attention',
            hparams)
        enc = feedforward(
            enc,
            weights,
            block_prefix + '/feedforward',
            hparams)

    hidden = dense(enc, weights, 'score_hidden', hparams['activation'])
    scores = np.squeeze(dense(hidden, weights, 'score'), axis=-1)
    return np.where(mask == 0.0, np.float32(hparams['empty_bound']), scores)


def predict(args):
    metadata = load_metadata(args.train_dir)
    hparams = parse_hparams(metadata.get('hparams', ''))
    weights_path = args.weights or os.path.join(args.train_dir, 'numpy_weights.npz')
    if not os.path.isfile(weights_path):
        raise ValueError('Missing NumPy weights file: %s' % weights_path)

    example = build_prediction_example(args, metadata)
    normalizer = data_utils.csp_feature_normalizer_from_stats(
        metadata['instance_feature_mean'],
        metadata['instance_feature_std'])
    batch = data_utils.make_csp_score_batch(
        [example],
        pad_id=len(metadata['propagators']),
        max_prop_count=metadata['rank_list_size'],
        feature_normalizer=normalizer)

    with np.load(weights_path) as loaded:
        weights = dict((key, loaded[key]) for key in loaded.files)
    raw_scores = score_batch(weights, batch, hparams)[0]

    propagators = example['propagators']
    score_items = [(name, float(raw_scores[index])) for index, name in enumerate(propagators)]
    groups, priorities = group_scores_into_priorities(score_items)

    return {
        'instance': example['instance'],
        'checkpoint': weights_path,
        'scores': dict(score_items),
        'ordered_propagators': [
            name for name, _ in sorted(score_items, key=lambda item: (item[1], item[0]))],
        'priority_groups': [
            {
                'priority': CHOCO_PRIORITIES[index],
                'propagators': [name for name, _ in group],
            }
            for index, group in enumerate(groups)
        ],
        'priorities': priorities,
    }


def main():
    args = parse_args()
    result = predict(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        with open(args.output, 'w') as fout:
            fout.write(text)
            fout.write('\n')


if __name__ == '__main__':
    main()
