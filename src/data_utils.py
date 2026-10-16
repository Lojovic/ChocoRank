import numpy as np
import json
import random
import os
import hashlib
import csv
from collections import OrderedDict


class CspRuntimeData:
    def __init__(self, records=None, tie_seed=0, instance_features=None,
                 instance_feature_names=None, min_pair_runtime_ratio=0.0,
                 pair_sampling_strategy='weighted',
                 pair_keep_floor=0.1,
                 pair_sampling_seed=0,
                 pair_top_k=3,
                 timeout_ms=20 * 60 * 1000,
                 timeout_penalty_multiplier=1.0):
        self.tie_seed = tie_seed
        self.min_pair_runtime_ratio = min_pair_runtime_ratio
        self.pair_sampling_strategy = pair_sampling_strategy
        self.pair_keep_floor = pair_keep_floor
        self.pair_sampling_seed = pair_sampling_seed
        self.pair_top_k = pair_top_k
        self.timeout_ms = float(timeout_ms)
        self.timeout_penalty_multiplier = float(timeout_penalty_multiplier)
        self.instance_feature_map = instance_features or {}
        self.instance_feature_names = instance_feature_names or []
        self.instance_feature_size = len(self.instance_feature_names)
        self.propagators = []
        self.prop_to_id = {}
        self.instances = []
        self.runs = []
        self.runs_by_instance = OrderedDict()
        self.examples = []
        self.example_by_instance = OrderedDict()
        self.rank_list_size = 0
        self.pair_count = 0

        if records is not None:
            self._build(records)

    def _build(self, records):
        propagator_names = set()
        for row_index, record in enumerate(records):
            self._validate_record(record, row_index)
            propagator_names.update(record['propagator_scores'].keys())

        self.propagators = sorted(propagator_names)
        self.prop_to_id = dict((name, index) for index, name in enumerate(self.propagators))

        for row_index, record in enumerate(records):
            instance = record['instance']
            raw_runtime_ms = float(record['runtime_ms'])
            scores = dict((name, float(score))
                          for name, score in record['propagator_scores'].items())
            ordering = self._ordering_from_scores(instance, row_index, scores)
            run = {
                'row_index': row_index,
                'instance': instance,
                'raw_runtime_ms': raw_runtime_ms,
                'runtime_ms': self._effective_runtime_ms(raw_runtime_ms),
                'propagator_scores': scores,
                'ordering': ordering,
                'ordering_ids': [self.prop_to_id[name] for name in ordering],
            }

            self.runs.append(run)
            if instance not in self.runs_by_instance:
                self.runs_by_instance[instance] = []
            self.runs_by_instance[instance].append(run)
            self.rank_list_size = max(self.rank_list_size, len(ordering))

        self.instances = list(self.runs_by_instance.keys())
        self._build_instance_examples()

    def _validate_record(self, record, row_index):
        required_fields = ['instance', 'runtime_ms', 'propagator_scores']
        for field in required_fields:
            if field not in record:
                raise ValueError('CSP dataset row %d is missing field %s' % (row_index, field))
        if not isinstance(record['propagator_scores'], dict):
            raise ValueError('CSP dataset row %d propagator_scores must be an object' % row_index)
        if len(record['propagator_scores']) == 0:
            raise ValueError('CSP dataset row %d has no propagator scores' % row_index)

    def _ordering_from_scores(self, instance, row_index, scores):
        score_groups = OrderedDict()
        for propagator, score in sorted(scores.items(), key=lambda item: (item[1], item[0])):
            if score not in score_groups:
                score_groups[score] = []
            score_groups[score].append(propagator)

        ordering = []
        for score, group in score_groups.items():
            if len(group) > 1:
                rng = random.Random(self._tie_seed(instance, row_index, score, group))
                rng.shuffle(group)
            ordering.extend(group)
        return ordering

    def _tie_seed(self, instance, row_index, score, group):
        payload = '%s|%s|%s|%s|%s' % (
            self.tie_seed,
            instance,
            row_index,
            score,
            ','.join(sorted(group)),
        )
        return int(hashlib.md5(payload.encode('utf-8')).hexdigest()[:16], 16)

    def _build_instance_examples(self):
        for instance, runs in self.runs_by_instance.items():
            propagators = sorted(runs[0]['propagator_scores'].keys())
            expected = set(propagators)
            for run in runs:
                actual = set(run['propagator_scores'].keys())
                if actual != expected:
                    raise ValueError(
                        'Runs for instance %s do not share the same propagator set' % instance)

            local_prop_index = dict((name, index) for index, name in enumerate(propagators))
            example_runs = []
            for run_index, run in enumerate(runs):
                example_runs.append({
                    'run_index': run_index,
                    'row_index': run['row_index'],
                    'raw_runtime_ms': run['raw_runtime_ms'],
                    'runtime_ms': run['runtime_ms'],
                    'propagator_scores': run['propagator_scores'],
                    'ordering': run['ordering'],
                    'ordering_ids': run['ordering_ids'],
                    'ordering_indices': [local_prop_index[name] for name in run['ordering']],
                })

            pairs = self._runtime_pairs(instance, example_runs)
            example = {
                'instance': instance,
                'instance_features': self._features_for_instance(instance),
                'propagators': propagators,
                'propagator_ids': [self.prop_to_id[name] for name in propagators],
                'runs': example_runs,
                'pairs': pairs,
            }

            self.examples.append(example)
            self.example_by_instance[instance] = example
            self.pair_count += len(pairs)

    def _effective_runtime_ms(self, runtime_ms):
        if runtime_ms >= self.timeout_ms:
            return self.timeout_penalty_multiplier * self.timeout_ms
        return runtime_ms

    def _runtime_pairs(self, instance, runs):
        if self.pair_sampling_strategy == 'topk_bottomk':
            return self._topk_bottomk_runtime_pairs(runs)

        pairs = []
        for left in range(len(runs)):
            for right in range(left + 1, len(runs)):
                pair = self._make_runtime_pair(runs, left, right)
                if pair is None:
                    continue
                if not self._keep_runtime_pair(instance, runs, left, right,
                                               pair['runtime_delta_ratio']):
                    continue
                pairs.append(pair)
        return pairs

    def _topk_bottomk_runtime_pairs(self, runs):
        k = max(int(self.pair_top_k), 1)
        ranked = sorted(
            range(len(runs)),
            key=lambda index: (runs[index]['runtime_ms'], runs[index]['row_index']))
        fastest = ranked[:k]
        slowest = ranked[-k:]
        pairs = []
        seen = set()
        for left in fastest:
            for right in slowest:
                if left == right:
                    continue
                key = tuple(sorted((left, right)))
                if key in seen:
                    continue
                seen.add(key)
                pair = self._make_runtime_pair(runs, left, right)
                if pair is not None:
                    pairs.append(pair)
        return pairs

    def _make_runtime_pair(self, runs, left, right):
        left_runtime = runs[left]['runtime_ms']
        right_runtime = runs[right]['runtime_ms']
        if left_runtime == right_runtime:
            return None
        if left_runtime < right_runtime:
            faster = left
            slower = right
            runtime_delta = right_runtime - left_runtime
            relative_delta = runtime_delta / max(left_runtime, 1.0)
        else:
            faster = right
            slower = left
            runtime_delta = left_runtime - right_runtime
            relative_delta = runtime_delta / max(right_runtime, 1.0)
        return {
            'faster': faster,
            'slower': slower,
            'runtime_delta_ms': runtime_delta,
            'runtime_delta_ratio': relative_delta,
        }

    def _keep_runtime_pair(self, instance, runs, left, right, relative_delta):
        if self.pair_sampling_strategy == 'all_pairs':
            return True
        if self.pair_sampling_strategy == 'threshold':
            return relative_delta >= self.min_pair_runtime_ratio
        if self.pair_sampling_strategy != 'weighted':
            raise ValueError('Unknown pair_sampling_strategy: %s' %
                             self.pair_sampling_strategy)

        full_keep_ratio = max(float(self.min_pair_runtime_ratio), 1e-12)
        keep_floor = min(max(float(self.pair_keep_floor), 0.0), 1.0)
        scaled_gap = min(max(float(relative_delta) / full_keep_ratio, 0.0), 1.0)
        keep_probability = keep_floor + (1.0 - keep_floor) * scaled_gap
        if keep_probability >= 1.0:
            return True

        payload = '%s|%s|%s|%s' % (
            self.pair_sampling_seed,
            instance,
            left,
            right,
        )
        value = int(hashlib.md5(payload.encode('utf-8')).hexdigest()[:16], 16)
        sample = float(value) / float(16 ** 16 - 1)
        return sample < keep_probability

    def _features_for_instance(self, instance):
        if not self.instance_feature_map:
            return []
        if instance not in self.instance_feature_map:
            raise ValueError('Missing instance features for %s' % instance)
        return self.instance_feature_map[instance]


def _read_json_records(json_path):
    with open(json_path) as fin:
        text = fin.read().strip()

    if not text:
        return []

    try:
        payload = json.loads(text)
    except ValueError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and 'records' in payload:
        return payload['records']
    raise ValueError('Expected %s to contain a JSON array, JSON lines, or an object with records' % json_path)


def read_instance_features_csv(csv_path, instance_column='instance_name'):
    with open(csv_path, newline='') as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise ValueError('Instance feature file %s has no header' % csv_path)
        if instance_column not in reader.fieldnames:
            raise ValueError(
                'Instance feature file %s is missing column %s' %
                (csv_path, instance_column))

        feature_names = [name for name in reader.fieldnames if name != instance_column]
        features = OrderedDict()
        for row_index, row in enumerate(reader):
            instance = row[instance_column]
            if instance in features:
                raise ValueError('Duplicate instance features for %s' % instance)
            try:
                features[instance] = [float(row[name]) for name in feature_names]
            except ValueError as exc:
                raise ValueError(
                    'Invalid numeric instance feature in row %d of %s: %s' %
                    (row_index + 2, csv_path, exc))
    return features, feature_names


def read_csp_runtime_data(json_path, tie_seed=0, instance_feature_path=None,
                          min_pair_runtime_ratio=0.0,
                          pair_sampling_strategy='weighted',
                          pair_keep_floor=0.1,
                          pair_sampling_seed=0,
                          pair_top_k=3,
                          timeout_ms=20 * 60 * 1000,
                          timeout_penalty_multiplier=1.0):
    instance_features = None
    instance_feature_names = None
    if instance_feature_path is not None:
        instance_features, instance_feature_names = read_instance_features_csv(
            instance_feature_path)
    return CspRuntimeData(
        _read_json_records(json_path),
        tie_seed=tie_seed,
        instance_features=instance_features,
        instance_feature_names=instance_feature_names,
        min_pair_runtime_ratio=min_pair_runtime_ratio,
        pair_sampling_strategy=pair_sampling_strategy,
        pair_keep_floor=pair_keep_floor,
        pair_sampling_seed=pair_sampling_seed,
        pair_top_k=pair_top_k,
        timeout_ms=timeout_ms,
        timeout_penalty_multiplier=timeout_penalty_multiplier)


class CspFeatureNormalizer:
    def __init__(self, mean, std, epsilon=1e-6):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.epsilon = epsilon
        self.std = np.where(self.std < epsilon, 1.0, self.std)

    @classmethod
    def fit(cls, examples, epsilon=1e-6):
        feature_rows = [example.get('instance_features', []) for example in examples]
        if not feature_rows or len(feature_rows[0]) == 0:
            return cls([], [], epsilon=epsilon)
        features = np.array(feature_rows, dtype=np.float32)
        return cls(np.mean(features, axis=0), np.std(features, axis=0), epsilon=epsilon)

    def transform(self, features):
        features = np.array(features, dtype=np.float32)
        if features.size == 0:
            return features
        return (features - self.mean) / self.std

    def transform_batch(self, feature_rows):
        features = np.array(feature_rows, dtype=np.float32)
        if features.size == 0:
            return features
        return (features - self.mean) / self.std


def fit_csp_feature_normalizer(examples, epsilon=1e-6):
    return CspFeatureNormalizer.fit(examples, epsilon=epsilon)


def csp_feature_normalizer_from_stats(mean, std, epsilon=1e-6):
    return CspFeatureNormalizer(mean, std, epsilon=epsilon)


def make_csp_score_batch(examples, pad_id, max_prop_count=None,
                         feature_normalizer=None):
    if max_prop_count is None:
        max_prop_count = max(len(example['propagator_ids']) for example in examples)

    propagator_ids = []
    propagator_mask = []
    instance_features = []
    instance_feature_size = len(examples[0].get('instance_features', []))
    for example in examples:
        ids = list(example['propagator_ids'])
        if len(ids) > max_prop_count:
            raise ValueError(
                'Instance %s has %d propagators, but max_prop_count is %d' %
                (example['instance'], len(ids), max_prop_count))

        pad_count = max_prop_count - len(ids)
        propagator_ids.append(ids + [pad_id] * pad_count)
        propagator_mask.append([1.0] * len(ids) + [0.0] * pad_count)
        features = list(example.get('instance_features', []))
        if len(features) != instance_feature_size:
            raise ValueError('Inconsistent instance feature size in batch')
        instance_features.append(features)

    if feature_normalizer is not None:
        instance_features = feature_normalizer.transform_batch(instance_features)

    return {
        'propagator_ids': np.array(propagator_ids, dtype=np.int64),
        'propagator_mask': np.array(propagator_mask, dtype=np.float32),
        'instance_features': np.array(instance_features, dtype=np.float32),
    }


def make_csp_prediction_example(instance, propagators, prop_to_id, instance_features):
    missing = [name for name in propagators if name not in prop_to_id]
    if missing:
        raise ValueError('Unknown propagators: %s' % ', '.join(sorted(missing)))
    return {
        'instance': instance,
        'instance_features': list(instance_features),
        'propagators': list(propagators),
        'propagator_ids': [prop_to_id[name] for name in propagators],
    }


def make_csp_training_batch(examples, pad_id, max_prop_count=None,
                            max_run_count=None, max_pair_count=None,
                            feature_normalizer=None):
    batch = make_csp_score_batch(
        examples,
        pad_id,
        max_prop_count=max_prop_count,
        feature_normalizer=feature_normalizer)
    max_prop_count = batch['propagator_ids'].shape[1]
    if max_run_count is None:
        max_run_count = max(len(example['runs']) for example in examples)
    if max_pair_count is None:
        max_pair_count = max(len(example['pairs']) for example in examples)

    orderings = []
    ordering_mask = []
    run_mask = []
    pair_faster = []
    pair_slower = []
    pair_weight = []
    pair_mask = []

    for example in examples:
        if len(example['runs']) > max_run_count:
            raise ValueError(
                'Instance %s has %d runs, but max_run_count is %d' %
                (example['instance'], len(example['runs']), max_run_count))
        if len(example['pairs']) > max_pair_count:
            raise ValueError(
                'Instance %s has %d pairs, but max_pair_count is %d' %
                (example['instance'], len(example['pairs']), max_pair_count))

        example_orderings = []
        example_ordering_mask = []
        for run in example['runs']:
            ordering = list(run['ordering_indices'])
            pad_count = max_prop_count - len(ordering)
            example_orderings.append(ordering + [0] * pad_count)
            example_ordering_mask.append([1.0] * len(ordering) + [0.0] * pad_count)

        run_pad_count = max_run_count - len(example_orderings)
        example_orderings += [[0] * max_prop_count for _ in range(run_pad_count)]
        example_ordering_mask += [[0.0] * max_prop_count for _ in range(run_pad_count)]
        orderings.append(example_orderings)
        ordering_mask.append(example_ordering_mask)
        run_mask.append([1.0] * len(example['runs']) + [0.0] * run_pad_count)

        faster = []
        slower = []
        weights = []
        for pair in example['pairs']:
            faster.append(pair['faster'])
            slower.append(pair['slower'])
            weights.append(np.log1p(float(pair['runtime_delta_ratio'])))
        pair_pad_count = max_pair_count - len(faster)
        pair_faster.append(faster + [0] * pair_pad_count)
        pair_slower.append(slower + [0] * pair_pad_count)
        pair_weight.append(weights + [0.0] * pair_pad_count)
        pair_mask.append([1.0] * len(faster) + [0.0] * pair_pad_count)

    batch.update({
        'orderings': np.array(orderings, dtype=np.int64),
        'ordering_mask': np.array(ordering_mask, dtype=np.float32),
        'run_mask': np.array(run_mask, dtype=np.float32),
        'pair_faster': np.array(pair_faster, dtype=np.int64),
        'pair_slower': np.array(pair_slower, dtype=np.int64),
        'pair_weight': np.array(pair_weight, dtype=np.float32),
        'pair_mask': np.array(pair_mask, dtype=np.float32),
    })
    return batch
