from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

if not hasattr(tf, 'placeholder'):
    tf = tf.compat.v1
    tf.disable_v2_behavior()


def _hparams(**kwargs):
    if hasattr(tf, 'contrib'):
        return tf.contrib.training.HParams(**kwargs)

    class HParams(object):
        def __init__(self, **values):
            for key, value in values.items():
                setattr(self, key, value)

        def parse(self, hparam_str):
            if not hparam_str:
                return
            for item in hparam_str.split(','):
                if not item:
                    continue
                key, value = item.split('=', 1)
                if not hasattr(self, key):
                    raise ValueError('Unknown hyperparameter %s' % key)
                old_value = getattr(self, key)
                if isinstance(old_value, bool):
                    value = value.lower() in ('true', '1', 'yes')
                else:
                    value = type(old_value)(value)
                setattr(self, key, value)

    return HParams(**kwargs)


def _xavier_initializer():
    if hasattr(tf, 'contrib'):
        return tf.contrib.layers.xavier_initializer()
    return tf.keras.initializers.GlorotUniform()


def _dense(inputs, units, activation=None, name=None):
    return tf.keras.layers.Dense(
        units, activation=activation, kernel_initializer=_xavier_initializer(),
        name=name)(inputs)


def _dropout(inputs, rate, training, name=None):
    return tf.keras.layers.Dropout(rate, name=name)(inputs, training=training)


def _layer_norm(inputs, name=None):
    return tf.keras.layers.LayerNormalization(epsilon=1e-8, name=name)(inputs)


def _feedforward(inputs, hidden_units, activation, scope):
    with tf.variable_scope(scope):
        outputs = _dense(inputs, 4 * hidden_units, activation=activation, name='inner')
        outputs = _dense(outputs, hidden_units, name='readout')
        return _layer_norm(outputs + inputs, name='ln')


def _multihead_attention_mask(queries, keys, query_masks, key_masks, hidden_units,
                              num_heads, dropout_rate, is_training, activation, scope):
    if hidden_units % num_heads != 0:
        raise ValueError('hidden_units must be divisible by num_heads')

    with tf.variable_scope(scope):
        q = _dense(queries, hidden_units, activation=activation, name='q')
        k = _dense(keys, hidden_units, activation=activation, name='k')
        v = _dense(keys, hidden_units, activation=activation, name='v')

        q_heads = tf.concat(tf.split(q, num_heads, axis=2), axis=0)
        k_heads = tf.concat(tf.split(k, num_heads, axis=2), axis=0)
        v_heads = tf.concat(tf.split(v, num_heads, axis=2), axis=0)

        logits = tf.matmul(q_heads, tf.transpose(k_heads, [0, 2, 1]))
        logits = logits / ((hidden_units // num_heads) ** 0.5)

        tiled_key_masks = tf.tile(key_masks, [num_heads, 1])
        tiled_key_masks = tf.tile(
            tf.expand_dims(tiled_key_masks, 1), [1, tf.shape(queries)[1], 1])
        paddings = tf.ones_like(logits) * np.float32(-2 ** 32 + 1)
        logits = tf.where(tf.equal(tiled_key_masks, 0.0), paddings, logits)

        weights = tf.nn.softmax(logits)
        tiled_query_masks = tf.tile(query_masks, [num_heads, 1])
        tiled_query_masks = tf.tile(
            tf.expand_dims(tiled_query_masks, -1), [1, 1, tf.shape(keys)[1]])
        weights *= tiled_query_masks
        weights = _dropout(weights, dropout_rate, is_training, name='attention_dropout')

        outputs = tf.matmul(weights, v_heads)
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        return _layer_norm(outputs + queries, name='ln')


class CspRank(object):
    def __init__(self, num_propagators, max_prop_count, batch_size, hparam_str='',
                 max_run_count=20, max_pair_count=190, instance_feature_size=0):
        self.hparams = _hparams(
            learning_rate=0.001,
            max_gradient_norm=5.0,
            pl_margin=1.0,
            prop_embedding_size=64,
            hidden_units=128,
            num_blocks=2,
            num_heads=4,
            dropout_rate=0.1,
            activation='relu',
            empty_bound=1e9,
        )
        self.hparams.parse(hparam_str)

        self.num_propagators = num_propagators
        self.pad_id = num_propagators
        self.max_prop_count = max_prop_count
        self.max_run_count = max_run_count
        self.max_pair_count = max_pair_count
        self.instance_feature_size = instance_feature_size
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(self.hparams.learning_rate), trainable=False)
        self.global_step = tf.Variable(0, trainable=False)

        hp = self.hparams

        if hp.activation == 'elu':
            activation = tf.nn.elu
        else:
            activation = tf.nn.relu

        self.propagator_ids = tf.placeholder(
            tf.int64, shape=[None, max_prop_count], name='propagator_ids')
        self.propagator_mask = tf.placeholder(
            tf.float32, shape=[None, max_prop_count], name='propagator_mask')
        self.instance_features = tf.placeholder(
            tf.float32, shape=[None, instance_feature_size], name='instance_features')
        self.orderings = tf.placeholder(
            tf.int64, shape=[None, max_run_count, max_prop_count], name='orderings')
        self.ordering_mask = tf.placeholder(
            tf.float32, shape=[None, max_run_count, max_prop_count], name='ordering_mask')
        self.run_mask = tf.placeholder(
            tf.float32, shape=[None, max_run_count], name='run_mask')
        self.pair_faster = tf.placeholder(
            tf.int64, shape=[None, max_pair_count], name='pair_faster')
        self.pair_slower = tf.placeholder(
            tf.int64, shape=[None, max_pair_count], name='pair_slower')
        self.pair_weight = tf.placeholder(
            tf.float32, shape=[None, max_pair_count], name='pair_weight')
        self.pair_mask = tf.placeholder(
            tf.float32, shape=[None, max_pair_count], name='pair_mask')

        prop_embeddings = tf.get_variable(
            'propagator_embeddings',
            shape=[num_propagators, hp.prop_embedding_size],
            dtype=tf.float32,
            initializer=_xavier_initializer())
        pad_embedding = tf.zeros([1, hp.prop_embedding_size], dtype=tf.float32)
        prop_embeddings_with_pad = tf.concat([prop_embeddings, pad_embedding], axis=0)

        enc = tf.nn.embedding_lookup(prop_embeddings_with_pad, self.propagator_ids)
        enc = _dense(enc, hp.hidden_units, activation=activation,
                     name='input_projection')
        if instance_feature_size > 0:
            instance_context = _dense(
                self.instance_features, hp.hidden_units, activation=activation,
                name='instance_projection')
            instance_context = tf.expand_dims(instance_context, axis=1)
            instance_context = tf.tile(
                instance_context, [1, max_prop_count, 1])
            enc = _dense(
                tf.concat([enc, instance_context], axis=2),
                hp.hidden_units,
                activation=activation,
                name='instance_conditioned_projection')
        enc = _dropout(enc, hp.dropout_rate, True, name='input_dropout')

        for block_index in range(hp.num_blocks):
            with tf.variable_scope('csp_rank_block_%d' % block_index):
                enc = _multihead_attention_mask(
                    queries=enc,
                    keys=enc,
                    query_masks=self.propagator_mask,
                    key_masks=self.propagator_mask,
                    hidden_units=hp.hidden_units,
                    num_heads=hp.num_heads,
                    dropout_rate=hp.dropout_rate,
                    is_training=True,
                    activation=activation,
                    scope='self_attention')
                enc = _feedforward(
                    enc,
                    hidden_units=hp.hidden_units,
                    activation=activation,
                    scope='feedforward')

        score_hidden = _dense(
            enc, hp.hidden_units, activation=activation, name='score_hidden')
        raw_scores = tf.squeeze(
            _dense(score_hidden, 1, name='score'), axis=-1)

        padding_scores = tf.ones_like(raw_scores) * np.float32(hp.empty_bound)
        self.scores = tf.where(
            tf.equal(self.propagator_mask, 0.0), padding_scores, raw_scores)

        self.run_log_probs = self._plackett_luce_log_probs(
            self.scores, self.orderings, self.ordering_mask)
        self.loss = self._pl_hinge_loss(self.run_log_probs)

        params = tf.trainable_variables()
        opt = tf.train.AdamOptimizer(self.learning_rate)
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(
            gradients, hp.max_gradient_norm)
        self.updates = opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables())

    def _gather_local_scores(self, scores, orderings):
        one_hot = tf.one_hot(orderings, self.max_prop_count, dtype=tf.float32)
        scores = tf.expand_dims(tf.expand_dims(scores, axis=1), axis=1)
        return tf.reduce_sum(one_hot * scores, axis=3)

    def _plackett_luce_log_probs(self, scores, orderings, ordering_mask):
        # Lower scores mean earlier priorities
        ordered_scores = self._gather_local_scores(-scores, orderings)
        terms = []
        for position in range(self.max_prop_count):
            remaining_scores = ordered_scores[:, :, position:]
            remaining_mask = ordering_mask[:, :, position:]
            masked_remaining_scores = tf.where(
                tf.equal(remaining_mask, 0.0),
                tf.ones_like(remaining_scores) * np.float32(-1e9),
                remaining_scores)
            denominator = tf.reduce_logsumexp(masked_remaining_scores, axis=2)
            term = ordered_scores[:, :, position] - denominator
            term = tf.where(
                tf.equal(ordering_mask[:, :, position], 0.0),
                tf.zeros_like(term),
                term)
            terms.append(term)
        log_probs = tf.add_n(terms)
        return log_probs * self.run_mask

    def _gather_run_values(self, values, indices):
        one_hot = tf.one_hot(indices, self.max_run_count, dtype=tf.float32)
        values = tf.expand_dims(values, axis=1)
        return tf.reduce_sum(one_hot * values, axis=2)

    def _pl_hinge_loss(self, run_log_probs):
        faster_log_probs = self._gather_run_values(run_log_probs, self.pair_faster)
        slower_log_probs = self._gather_run_values(run_log_probs, self.pair_slower)
        losses = tf.nn.relu(
            np.float32(self.hparams.pl_margin) + slower_log_probs - faster_log_probs)
        weighted_mask = self.pair_mask * self.pair_weight
        losses *= weighted_mask
        pair_weight_sum = tf.maximum(tf.reduce_sum(weighted_mask), np.float32(1.0))
        return tf.reduce_sum(losses) / pair_weight_sum

    def _training_feed(self, batch):
        return {
            self.propagator_ids.name: batch['propagator_ids'],
            self.propagator_mask.name: batch['propagator_mask'],
            self.instance_features.name: batch['instance_features'],
            self.orderings.name: batch['orderings'],
            self.ordering_mask.name: batch['ordering_mask'],
            self.run_mask.name: batch['run_mask'],
            self.pair_faster.name: batch['pair_faster'],
            self.pair_slower.name: batch['pair_slower'],
            self.pair_weight.name: batch['pair_weight'],
            self.pair_mask.name: batch['pair_mask'],
        }

    def evaluate_batch(self, session, batch):
        return session.run(
            [self.loss, self.run_log_probs, self.scores],
            self._training_feed(batch))

    def train_step(self, session, batch):
        outputs = session.run(
            [self.updates, self.gradient_norm, self.loss], self._training_feed(batch))
        return outputs[1], outputs[2]
