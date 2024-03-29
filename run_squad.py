# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

from __future__ import absolute_import, division, print_function

import collections
import json
import math
import os
import random
import shutil
import time

import horovod.tensorflow as hvd
import numpy as np
import six
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import device_lib

import modeling_relpo as modeling_func
import optimization
import tokenization
from utils.create_squad_data import *
from utils.utils import LogEvalRunHook, LogTrainRunHook

from qa_utils.loss_helper import (
    compute_double_forward_loss_w_add_noise,
    compute_vat_loss,
    compute_approx_jacobian_loss,
)


flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("train_file", None,
                    "SQuAD json for training. E.g., train-v1.1.json")

flags.DEFINE_string(
    "predict_file", None,
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 8, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-6, "The initial learning rate for Adam.")

flags.DEFINE_bool("use_trt", False, "Whether to use TF-TRT")

flags.DEFINE_bool("horovod", False, "Whether to use Horovod for multi-gpu runs")
flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_float(
    "freeze_proportion", 0.1,
    "Proportion of training to freeze certain weights. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_bool("freeze_bert", False, "Whether to first freeze bert weights.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("num_accumulation_steps", 1,
                     "Number of accumulation steps before gradient update"
                      "Global batch size = num_accumulation_steps * train_batch_size")

flags.DEFINE_string(
    "optimizer_type", "adam",
    "Optimizer used for training - LAMB or ADAM")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")


flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_bool(
    "version_2_with_negative", False,
    "If true, the SQuAD examples contain some that do not have an answer.")

flags.DEFINE_float(
    "null_score_diff_threshold", 0.0,
    "If null_score - best_non_null is greater than the threshold predict null.")

flags.DEFINE_bool("use_fp16", False, "Whether to use fp32 or fp16 arithmetic on GPU.")
flags.DEFINE_bool("manual_fp16", False, "Whether to use fp32 or fp16 arithmetic on GPU.")
flags.DEFINE_bool("use_xla", False, "Whether to enable XLA JIT compilation.")
flags.DEFINE_integer("num_eval_iterations", None,
                     "How many eval iterations to run - performs inference on subset")

# TRTIS Specific flags
flags.DEFINE_bool("export_trtis", False, "Whether to export saved model or run inference with TRTIS")
flags.DEFINE_string("trtis_model_name", "bert", "exports to appropriate directory for TRTIS")
flags.DEFINE_integer("trtis_model_version", 1, "exports to appropriate directory for TRTIS")
flags.DEFINE_string("trtis_server_url", "localhost:8001", "exports to appropriate directory for TRTIS")
flags.DEFINE_bool("trtis_model_overwrite", False, "If True, will overwrite an existing directory with the specified 'model_name' and 'version_name'")
flags.DEFINE_integer("trtis_max_batch_size", 8, "Specifies the 'max_batch_size' in the TRTIS model config. See the TRTIS documentation for more info.")
flags.DEFINE_float("trtis_dyn_batching_delay", 0, "Determines the dynamic_batching queue delay in milliseconds(ms) for the TRTIS model config. Use '0' or '-1' to specify static batching. See the TRTIS documentation for more info.")
flags.DEFINE_integer("trtis_engine_count", 1, "Specifies the 'instance_group' count value in the TRTIS model config. See the TRTIS documentation for more info.")

# Robustness-related flags.
flags.DEFINE_float(
    "double_forward_reg_rate", 0.0,
    "Weight for controlling the double forward loss."
)

flags.DEFINE_string(
    "double_forward_loss", "v2",
    "Double forward loss type, {v1, v2, v3}"
)

flags.DEFINE_float(
    "noise_epsilon", 1e-3,
    "Float for the random noise scale"
)

flags.DEFINE_string(
    "noise_normalizer", "L2",
    "String for the noise normalizer type {L2, L1, Linf}"
)

flags.DEFINE_float(
    "kl_alpha", 1e-3,
    "Float for scaling the entropy term of KL"
)

flags.DEFINE_float(
    "kl_beta", 1.0,
    "Float for scaling the cross entropy term of KL"
)

flags.DEFINE_float(
    "jacobian_reg_rate", 0.0,
    "Weight for controlling the jacobian loss.")

flags.DEFINE_float(
    "vat_reg_rate", 0.0,
    "Weight for controlling the vat loss.")

flags.DEFINE_float(
    "mlm_rate", 0.0,
    "Weight for controlling the maksed lm loss.")


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings):
  """Creates a classification model."""
  if bert_config.electra:
      scope = "electra"
  else:
      scope = "bert"

  model = modeling_func.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings,
      compute_type=tf.float16 if FLAGS.manual_fp16 else tf.float32,
      scope=scope
  )

  final_hidden = model.get_sequence_output()

  final_hidden_shape = modeling_func.get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]
  hidden_size = final_hidden_shape[2]

  output_weights = tf.get_variable(
      "cls/squad/output_weights", [2, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

  final_hidden_matrix = tf.reshape(final_hidden,
                                   [batch_size * seq_length, hidden_size])
  logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)

  logits = tf.reshape(logits, [batch_size, seq_length, 2])
  logits = tf.transpose(logits, [2, 0, 1])

  unstacked_logits = tf.unstack(logits, axis=0, name='unstack')

  (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

  start_logits = tf.cast(start_logits, dtype=tf.float32)
  end_logits = tf.cast(end_logits, dtype=tf.float32)

  return (start_logits, end_logits, model)

def get_frozen_tftrt_model(bert_config, shape, use_one_hot_embeddings, init_checkpoint):
  tf_config = tf.ConfigProto()
  output_node_names = ['unstack']

  with tf.Session(config=tf_config) as tf_sess:
    input_ids = tf.placeholder(tf.int32, shape, 'input_ids')
    input_mask = tf.placeholder(tf.int32, shape, 'input_mask')
    segment_ids = tf.placeholder(tf.int32, shape, 'segment_ids')

    (start_logits, end_logits) = create_model(bert_config=bert_config,
                                              is_training=False,
                                              input_ids=input_ids,
                                              input_mask=input_mask,
                                              segment_ids=segment_ids,
                                              use_one_hot_embeddings=use_one_hot_embeddings)


    tvars = tf.trainable_variables()
    (assignment_map, initialized_variable_names) = modeling_func.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    tf_sess.run(tf.global_variables_initializer())
    print("LOADED!")
    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      else:
        init_string = ", *NOTTTTTTTTTTTTTTTTTTTTT"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

    frozen_graph = tf.graph_util.convert_variables_to_constants(tf_sess,
            tf_sess.graph.as_graph_def(), output_node_names)

    num_nodes = len(frozen_graph.node)
    print('Converting graph using TensorFlow-TensorRT...')
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    converter = trt.TrtGraphConverter(
        input_graph_def=frozen_graph,
        nodes_blacklist=output_node_names,
        max_workspace_size_bytes=(4096 << 20) - 1000,
        precision_mode = "FP16" if FLAGS.use_fp16 else "FP32",
        minimum_segment_size=4,
        is_dynamic_op=True,
        maximum_cached_engines=1000
    )
    frozen_graph = converter.convert()

    print('Total node count before and after TF-TRT conversion:',
          num_nodes, '->', len(frozen_graph.node))
    print('TRT node count:',
          len([1 for n in frozen_graph.node if str(n.op) == 'TRTEngineOp']))

    with tf.gfile.GFile("frozen_modelTRT.pb", "wb") as f:
      f.write(frozen_graph.SerializeToString())

  return frozen_graph

def create_reuse_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                       use_one_hot_embeddings):
  """Creates a classification model."""
  if bert_config.electra:
      scope = "electra"
  else:
      scope = "bert"

  model = modeling_func.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings,
      compute_type=tf.float16 if FLAGS.manual_fp16 else tf.float32,
      scope=scope,
      reuse=True
  )

  return model


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return tf.reshape(output_tensor, [batch_size, -1, width])


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM.
  Args:
    bert_config: BertConfig object contains model architecture info.
    input_tensor: A [batch_size, max_seq_len, hidden_dim]-sized feature tensor.
    output_weights: A [vocab_size, word_embed_dim]-sized embedding weight.
    label_ids: A [batch_size, max_seq_len]-sized index tensor.
    label_weights: A [batch_size, max_seq_len]-sized binary tensor.
  """
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(tf.cast(input_tensor, tf.float32), output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # label_ids = tf.reshape(label_ids, [-1])
    # label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=-1)
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return loss


Inputs = collections.namedtuple(
    "Inputs", ["input_ids", "input_mask", "segment_ids", "masked_lm_positions",
               "masked_lm_ids", "masked_lm_weights"])


def features_to_inputs(features):
  return Inputs(
      input_ids=features["input_ids"],
      input_mask=features["input_mask"],
      segment_ids=features["segment_ids"],
      masked_lm_positions=(features["masked_lm_positions"]
                           if "masked_lm_positions" in features else None),
      masked_lm_ids=(features["masked_lm_ids"]
                     if "masked_lm_ids" in features else None),
      masked_lm_weights=(features["masked_lm_weights"]
                         if "masked_lm_weights" in features else None),
  )


def get_updated_inputs(inputs, **kwargs):
  features = inputs._asdict()
  for k, v in kwargs.items():
    features[k] = v
  return features_to_inputs(features)


def _get_candidates_mask(inputs: Inputs, vocab,
                         disallow_from_mask=None):
  """Returns a mask tensor of positions in the input that can be masked out."""
  ignore_ids = [vocab["[SEP]"], vocab["[CLS]"], vocab["[MASK]"]]
  candidates_mask = tf.ones_like(inputs.input_ids, tf.bool)
  for ignore_id in ignore_ids:
    candidates_mask &= tf.not_equal(inputs.input_ids, ignore_id)
  candidates_mask &= tf.cast(inputs.input_mask, tf.bool)
  if disallow_from_mask is not None:
    candidates_mask &= ~disallow_from_mask
  return candidates_mask


def scatter_update(sequence, updates, positions):
  """Scatter-update a sequence.

  Args:
    sequence: A [batch_size, seq_len] or [batch_size, seq_len, depth] tensor
    updates: A tensor of size batch_size*seq_len(*depth)
    positions: A [batch_size, n_positions] tensor

  Returns: A tuple of two tensors. First is a [batch_size, seq_len] or
    [batch_size, seq_len, depth] tensor of "sequence" with elements at
    "positions" replaced by the values at "updates." Updates to index 0 are
    ignored. If there are duplicated positions the update is only applied once.
    Second is a [batch_size, seq_len] mask tensor of which inputs were updated.
  """
  shape = modeling.get_shape_list(sequence, expected_rank=[2, 3])
  depth_dimension = (len(shape) == 3)
  if depth_dimension:
    B, L, D = shape
  else:
    B, L = shape
    D = 1
    sequence = tf.expand_dims(sequence, -1)
  N = modeling.get_shape_list(positions)[1]

  shift = tf.expand_dims(L * tf.range(B), -1)
  flat_positions = tf.reshape(positions + shift, [-1, 1])
  flat_updates = tf.reshape(updates, [-1, D])
  updates = tf.scatter_nd(flat_positions, flat_updates, [B * L, D])
  updates = tf.reshape(updates, [B, L, D])

  flat_updates_mask = tf.ones([B * N], tf.int32)
  updates_mask = tf.scatter_nd(flat_positions, flat_updates_mask, [B * L])
  updates_mask = tf.reshape(updates_mask, [B, L])
  not_first_token = tf.concat([tf.zeros((B, 1), tf.int32),
                               tf.ones((B, L - 1), tf.int32)], -1)
  updates_mask *= not_first_token
  updates_mask_3d = tf.expand_dims(updates_mask, -1)

  # account for duplicate positions
  if sequence.dtype == tf.float32:
    updates_mask_3d = tf.cast(updates_mask_3d, tf.float32)
    updates /= tf.maximum(1.0, updates_mask_3d)
  else:
    assert sequence.dtype == tf.int32
    updates = tf.math.floordiv(updates, tf.maximum(1, updates_mask_3d))
  updates_mask = tf.minimum(updates_mask, 1)
  updates_mask_3d = tf.minimum(updates_mask_3d, 1)

  updated_sequence = (((1 - updates_mask_3d) * sequence) +
                      (updates_mask_3d * updates))
  if not depth_dimension:
    updated_sequence = tf.squeeze(updated_sequence, -1)

  return updated_sequence, updates_mask



def mask(inputs: Inputs, mask_prob, vocab_file, do_lower_case,
         proposal_distribution=1.0,
         max_predictions_per_seq=80,
         disallow_from_mask=None, already_masked=None):
  """Implementation of dynamic masking. The optional arguments aren't needed for
  BERT/ELECTRA and are from early experiments in "strategically" masking out
  tokens instead of uniformly at random.

  Args:
    config: configure_pretraining.PretrainingConfig
    inputs: pretrain_data.Inputs containing input input_ids/input_mask
    mask_prob: percent of tokens to mask
    proposal_distribution: for non-uniform masking can be a [B, L] tensor
                           of scores for masking each position.
    disallow_from_mask: a boolean tensor of [B, L] of positions that should
                        not be masked out
    already_masked: a boolean tensor of [B, N] of already masked-out tokens
                    for multiple rounds of masking
  Returns: a pretrain_data.Inputs with masking added
  """
  # Get the batch size, sequence length, and max masked-out tokens
  N = max_predictions_per_seq
  B, L = modeling.get_shape_list(inputs.input_ids)

  # Find indices where masking out a token is allowed
  vocab = tokenization.FullTokenizer(
      vocab_file, do_lower_case=do_lower_case).vocab
  candidates_mask = _get_candidates_mask(inputs, vocab, disallow_from_mask)

  # Set the number of tokens to mask out per example
  num_tokens = tf.cast(tf.reduce_sum(inputs.input_mask, -1), tf.float32)
  num_to_predict = tf.maximum(1, tf.minimum(
      N, tf.cast(tf.round(num_tokens * mask_prob), tf.int32)))
  masked_lm_weights = tf.cast(tf.sequence_mask(num_to_predict, N), tf.float32)
  if already_masked is not None:
    masked_lm_weights *= (1 - already_masked)

  # Get a probability of masking each position in the sequence
  candidate_mask_float = tf.cast(candidates_mask, tf.float32)
  sample_prob = (proposal_distribution * candidate_mask_float)
  sample_prob /= tf.reduce_sum(sample_prob, axis=-1, keepdims=True)

  # Sample the positions to mask out
  sample_prob = tf.stop_gradient(sample_prob)
  sample_logits = tf.log(sample_prob)
  masked_lm_positions = tf.random.categorical(
      sample_logits, N, dtype=tf.int32)
  masked_lm_positions *= tf.cast(masked_lm_weights, tf.int32)

  # Get the ids of the masked-out tokens
  shift = tf.expand_dims(L * tf.range(B), -1)
  flat_positions = tf.reshape(masked_lm_positions + shift, [-1, 1])
  masked_lm_ids = tf.gather_nd(tf.reshape(inputs.input_ids, [-1]),
                               flat_positions)
  masked_lm_ids = tf.reshape(masked_lm_ids, [B, -1])
  masked_lm_ids *= tf.cast(masked_lm_weights, tf.int32)

  # Update the input ids
  replace_with_mask_positions = masked_lm_positions * tf.cast(
      tf.less(tf.random.uniform([B, N]), 0.85), tf.int32)
  inputs_ids, _ = scatter_update(
      inputs.input_ids, tf.fill([B, N], vocab["[MASK]"]),
      replace_with_mask_positions)

  return get_updated_inputs(
      inputs,
      input_ids=tf.stop_gradient(inputs_ids),
      masked_lm_positions=masked_lm_positions,
      masked_lm_ids=masked_lm_ids,
      masked_lm_weights=masked_lm_weights
  )


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps,
                     hvd=None, use_fp16=False, use_one_hot_embeddings=False):
  """Returns `model_fn` closure for Estimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for Estimator."""
    if FLAGS.verbose_logging:
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
          tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    if not is_training and FLAGS.use_trt:
        trt_graph = get_frozen_tftrt_model(bert_config, input_ids.shape, use_one_hot_embeddings, init_checkpoint)
        (start_logits, end_logits) = tf.import_graph_def(trt_graph,
                input_map={'input_ids':input_ids, 'input_mask':input_mask, 'segment_ids':segment_ids},
                return_elements=['unstack:0', 'unstack:1'],
                name='')
        predictions = {
            "unique_ids": unique_ids,
            "start_logits": start_logits,
            "end_logits": end_logits,
        }
        output_spec = tf.estimator.TPUEstimatorSpec(
            mode=mode, predictions=predictions)
        return output_spec

    (start_logits, end_logits, model) = create_model(
        bert_config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    if FLAGS.mlm_rate > 0.0 and is_training:
        # Mask the input
        masked_inputs = mask(
            features_to_inputs(features), 0.15, FLAGS.vocab_file,
            FLAGS.do_lower_case, max_predictions_per_seq=80)
        reuse_model = create_reuse_model(
            bert_config, is_training, masked_inputs.input_ids,
            masked_inputs.input_mask, masked_inputs.segment_ids,
            use_one_hot_embeddings)
        mlm_loss = get_masked_lm_output(
             bert_config,
             reuse_model.get_sequence_output(),
             reuse_model.get_embedding_table(),
             masked_inputs.masked_lm_positions,
             masked_inputs.masked_lm_ids,
             masked_inputs.masked_lm_weights
        )

    if bert_config.use_rel_pos_embeddings:
      tf.logging.info("***Uses relative position embeddings***")

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    if init_checkpoint and (hvd is None or hvd.rank() == 0):
      (assignment_map, initialized_variable_names
      ) = modeling_func.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    if FLAGS.verbose_logging or True:
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
          init_string = ""
          if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
          tf.logging.info(" %d name = %s, shape = %s%s", 0 if hvd is None else hvd.rank(), var.name, var.shape,
                          init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      seq_length = modeling_func.get_shape_list(input_ids)[1]

      def compute_loss(logits, positions):
        one_hot_positions = tf.one_hot(
            positions, depth=seq_length, dtype=tf.float32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(
            tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
        return loss

      start_positions = features["start_positions"]
      end_positions = features["end_positions"]

      start_logits = tf.cast(start_logits, tf.float32)
      end_logits = tf.cast(end_logits, tf.float32)

      start_loss = compute_loss(start_logits, start_positions)
      end_loss = compute_loss(end_logits, end_positions)

      clean_loss = (start_loss + end_loss) / 2.0
      total_loss = (start_loss + end_loss) / 2.0

      mlm_loss = tf.constant(0.0)

      if FLAGS.vat_reg_rate > 0.0:
          k = FLAGS.n_best_size
          loss_items = FLAGS.double_forward_loss.split("-")
          if len(loss_items) == 2:
              vat_type = loss_items[0]
              loss_type = loss_items[1]
          elif len(loss_items) == 1:
              vat_type = "vanilla"
              loss_type = loss_items[0]
          else:
              raise ValueError(
                  "Unknown loss config for vat %s" % FLAGS.double_forward_loss)

          tf.logging.info("Using vanilla VAT")
          vat_loss = compute_vat_loss(
              start_logits, end_logits, model,
              noise_epsilon=FLAGS.noise_epsilon,
              noise_normalizer=FLAGS.noise_normalizer,
              loss_type=loss_type,
          )
          total_loss += FLAGS.vat_reg_rate * vat_loss
      else:
          vat_loss = tf.constant(0.0)

      if FLAGS.jacobian_reg_rate > 0.0:
          jacobian_loss = compute_approx_jacobian_loss(
              start_logits, end_logits, model.get_embedding_output())
          total_loss += FLAGS.jacobian_reg_rate * jacobian_loss
      else:
          jacobian_loss = tf.constant(0.0)

      if FLAGS.double_forward_reg_rate > 0.0:
        double_forward_loss = compute_double_forward_loss_w_add_noise(
            start_logits, end_logits, model,
            loss_type=FLAGS.double_forward_loss,
            noise_epsilon=FLAGS.noise_epsilon,
            noise_normalizer=FLAGS.noise_normalizer,
            alpha=FLAGS.kl_alpha,
            beta=FLAGS.kl_beta,
        )


        total_loss += FLAGS.double_forward_reg_rate * double_forward_loss
      else:
        double_forward_loss = tf.constant(0.0)

      num_freeze_steps = 0
      non_freeze_scope = ["bert", "cls"]
      if FLAGS.freeze_bert:
        num_freeze_steps = int(num_train_steps * FLAGS.freeze_proportion)
        non_freeze_scope = ["cls"]

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, hvd=hvd,
          manual_fp16=FLAGS.manual_fp16, use_fp16=FLAGS.use_fp16,
          num_accumulation_steps=FLAGS.num_accumulation_steps,
          optimizer_type=FLAGS.optimizer_type,
          num_freeze_steps=num_freeze_steps,
          non_freeze_scope=non_freeze_scope,
      )

      logging_hook = tf.train.LoggingTensorHook(
          {"total_loss": total_loss, "clean_loss": clean_loss,
           "mlm_loss": mlm_loss,
           "scaled_mlm_loss": FLAGS.mlm_rate * mlm_loss,
           "jacobian_loss": jacobian_loss,
           "scaled_jacobian_loss": FLAGS.jacobian_reg_rate * jacobian_loss,
           "vat_loss": vat_loss,
           "scaled_vat_loss": FLAGS.vat_reg_rate * vat_loss,
           "double_forward_loss": double_forward_loss,
           "scaled_double_forward_loss": FLAGS.double_forward_reg_rate * double_forward_loss,
           },
          every_n_iter=1000)

      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          training_hooks=[logging_hook],
          train_op=train_op)
    elif mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          "unique_ids": unique_ids,
          "start_logits": start_logits,
          "end_logits": end_logits,
      }
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode, predictions=predictions)
    else:
      raise ValueError(
          "Only TRAIN and PREDICT modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def input_fn_builder(input_file, batch_size, seq_length, is_training, drop_remainder, hvd=None):
  """Creates an `input_fn` closure to be passed to Estimator."""

  name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
  }

  if is_training:
    name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn():
    """The actual input function."""

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
        d = tf.data.TFRecordDataset(input_file, num_parallel_reads=4)
        if hvd is not None: d = d.shard(hvd.size(), hvd.rank())
        d = d.apply(tf.data.experimental.ignore_errors())
        d = d.shuffle(buffer_size=100)
        d = d.repeat()
    else:
        d = tf.data.TFRecordDataset(input_file)


    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn



RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file):
  """Write final predictions to the json file and log-odds of null if needed."""
  tf.logging.info("Writing predictions to: %s" % (output_prediction_file))
  tf.logging.info("Writing nbest to: %s" % (output_nbest_file))

  example_index_to_features = collections.defaultdict(list)
  for feature in all_features:
    example_index_to_features[feature.example_index].append(feature)

  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
      "PrelimPrediction",
      ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

  all_predictions = collections.OrderedDict()
  all_nbest_json = collections.OrderedDict()
  scores_diff_json = collections.OrderedDict()

  for (example_index, example) in enumerate(all_examples):
    features = example_index_to_features[example_index]

    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    score_null = 1000000  # large and positive
    min_null_feature_index = 0  # the paragraph slice with min mull score
    null_start_logit = 0  # the start logit at the slice with min null score
    null_end_logit = 0  # the end logit at the slice with min null score
    for (feature_index, feature) in enumerate(features):
      result = unique_id_to_result[feature.unique_id]
      start_indexes = _get_best_indexes(result.start_logits, n_best_size)
      end_indexes = _get_best_indexes(result.end_logits, n_best_size)
      # if we could have irrelevant answers, get the min score of irrelevant
      if FLAGS.version_2_with_negative:
        feature_null_score = result.start_logits[0] + result.end_logits[0]
        if feature_null_score < score_null:
          score_null = feature_null_score
          min_null_feature_index = feature_index
          null_start_logit = result.start_logits[0]
          null_end_logit = result.end_logits[0]
      for start_index in start_indexes:
        for end_index in end_indexes:
          # We could hypothetically create invalid predictions, e.g., predict
          # that the start of the span is in the question. We throw out all
          # invalid predictions.
          if start_index >= len(feature.tokens):
            continue
          if end_index >= len(feature.tokens):
            continue
          if start_index not in feature.token_to_orig_map:
            continue
          if end_index not in feature.token_to_orig_map:
            continue
          if not feature.token_is_max_context.get(start_index, False):
            continue
          if end_index < start_index:
            continue
          length = end_index - start_index + 1
          if length > max_answer_length:
            continue
          prelim_predictions.append(
              _PrelimPrediction(
                  feature_index=feature_index,
                  start_index=start_index,
                  end_index=end_index,
                  start_logit=result.start_logits[start_index],
                  end_logit=result.end_logits[end_index]))

    if FLAGS.version_2_with_negative:
      prelim_predictions.append(
          _PrelimPrediction(
              feature_index=min_null_feature_index,
              start_index=0,
              end_index=0,
              start_logit=null_start_logit,
              end_logit=null_end_logit))
    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_logit", "end_logit"])

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:
        break
      feature = features[pred.feature_index]
      if pred.start_index > 0:  # this is a non-null prediction
        tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
        orig_doc_start = feature.token_to_orig_map[pred.start_index]
        orig_doc_end = feature.token_to_orig_map[pred.end_index]
        orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
        tok_text = " ".join(tok_tokens)

        # De-tokenize WordPieces that have been split off.
        tok_text = tok_text.replace(" ##", "")
        tok_text = tok_text.replace("##", "")

        # Clean whitespace
        tok_text = tok_text.strip()
        tok_text = " ".join(tok_text.split())
        orig_text = " ".join(orig_tokens)

        final_text = get_final_text(tok_text, orig_text, do_lower_case)
        if final_text in seen_predictions:
          continue

        seen_predictions[final_text] = True
      else:
        final_text = ""
        seen_predictions[final_text] = True

      nbest.append(
          _NbestPrediction(
              text=final_text,
              start_logit=pred.start_logit,
              end_logit=pred.end_logit))

    # if we didn't inlude the empty option in the n-best, inlcude it
    if FLAGS.version_2_with_negative:
      if "" not in seen_predictions:
        nbest.append(
            _NbestPrediction(
                text="", start_logit=null_start_logit,
                end_logit=null_end_logit))
    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
      nbest.append(
          _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

    assert len(nbest) >= 1

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
      total_scores.append(entry.start_logit + entry.end_logit)
      if not best_non_null_entry:
        if entry.text:
          best_non_null_entry = entry

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_logit"] = entry.start_logit
      output["end_logit"] = entry.end_logit
      nbest_json.append(output)

    assert len(nbest_json) >= 1

    if not FLAGS.version_2_with_negative:
      all_predictions[example.qas_id] = nbest_json[0]["text"]
    elif best_non_null_entry:
      # predict "" iff the null score - the score of best non-null > threshold
      score_diff = score_null - best_non_null_entry.start_logit - (
          best_non_null_entry.end_logit)
      scores_diff_json[example.qas_id] = score_diff
      if score_diff > FLAGS.null_score_diff_threshold:
        all_predictions[example.qas_id] = ""
      else:
        all_predictions[example.qas_id] = best_non_null_entry.text
    else:
      all_predictions[example.qas_id] = ""

    all_nbest_json[example.qas_id] = nbest_json

  with tf.gfile.GFile(output_prediction_file, "w") as writer:
    writer.write(json.dumps(all_predictions, indent=4) + "\n")

  with tf.gfile.GFile(output_nbest_file, "w") as writer:
    writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

  if FLAGS.version_2_with_negative:
    with tf.gfile.GFile(output_null_log_odds_file, "w") as writer:
      writer.write(json.dumps(scores_diff_json, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case):
  """Project the tokenized prediction back to the original text."""

  # When we created the data, we kept track of the alignment between original
  # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
  # now `orig_text` contains the span of our original text corresponding to the
  # span that we predicted.
  #
  # However, `orig_text` may contain extra characters that we don't want in
  # our prediction.
  #
  # For example, let's say:
  #   pred_text = steve smith
  #   orig_text = Steve Smith's
  #
  # We don't want to return `orig_text` because it contains the extra "'s".
  #
  # We don't want to return `pred_text` because it's already been normalized
  # (the SQuAD eval script also does punctuation stripping/lower casing but
  # our tokenizer does additional normalization like stripping accent
  # characters).
  #
  # What we really want to return is "Steve Smith".
  #
  # Therefore, we have to apply a semi-complicated alignment heruistic between
  # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
  # can fail in certain cases in which case we just return `orig_text`.

  def _strip_spaces(text):
    ns_chars = []
    ns_to_s_map = collections.OrderedDict()
    for (i, c) in enumerate(text):
      if c == " ":
        continue
      ns_to_s_map[len(ns_chars)] = i
      ns_chars.append(c)
    ns_text = "".join(ns_chars)
    return (ns_text, ns_to_s_map)

  # We first tokenize `orig_text`, strip whitespace from the result
  # and `pred_text`, and check if they are the same length. If they are
  # NOT the same length, the heuristic has failed. If they are the same
  # length, we assume the characters are one-to-one aligned.
  tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

  tok_text = " ".join(tokenizer.tokenize(orig_text))

  start_position = tok_text.find(pred_text)
  if start_position == -1:
    if FLAGS.verbose_logging:
      tf.logging.info(
          "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
    return orig_text
  end_position = start_position + len(pred_text) - 1

  (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
  (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

  if len(orig_ns_text) != len(tok_ns_text):
    if FLAGS.verbose_logging:
      tf.logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                      orig_ns_text, tok_ns_text)
    return orig_text

  # We then project the characters in `pred_text` back to `orig_text` using
  # the character-to-character alignment.
  tok_s_to_ns_map = {}
  for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
    tok_s_to_ns_map[tok_index] = i

  orig_start_position = None
  if start_position in tok_s_to_ns_map:
    ns_start_position = tok_s_to_ns_map[start_position]
    if ns_start_position in orig_ns_to_s_map:
      orig_start_position = orig_ns_to_s_map[ns_start_position]

  if orig_start_position is None:
    if FLAGS.verbose_logging:
      tf.logging.info("Couldn't map start position")
    return orig_text

  orig_end_position = None
  if end_position in tok_s_to_ns_map:
    ns_end_position = tok_s_to_ns_map[end_position]
    if ns_end_position in orig_ns_to_s_map:
      orig_end_position = orig_ns_to_s_map[ns_end_position]

  if orig_end_position is None:
    if FLAGS.verbose_logging:
      tf.logging.info("Couldn't map end position")
    return orig_text

  output_text = orig_text[orig_start_position:(orig_end_position + 1)]
  return output_text


def _get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs



def validate_flags_or_throw(bert_config):
  """Validate the input FLAGS or throw an exception."""
  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_predict and not FLAGS.export_trtis:
    raise ValueError("At least one of `do_train` or `do_predict` or `export_SavedModel` must be True.")

  if FLAGS.do_train:
    if not FLAGS.train_file:
      raise ValueError(
          "If `do_train` is True, then `train_file` must be specified.")
  if FLAGS.do_predict:
    if not FLAGS.predict_file:
      raise ValueError(
          "If `do_predict` is True, then `predict_file` must be specified.")

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_query_length "
        "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))


def export_model(estimator, export_dir, init_checkpoint):
    """Exports a checkpoint in SavedModel format in a directory structure compatible with TRTIS."""


    def serving_input_fn():
        label_ids = tf.placeholder(tf.int32, [None,], name='unique_ids')
        input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_ids')
        input_mask = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_mask')
        segment_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='segment_ids')
        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
            'unique_ids': label_ids,
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
        })()
        return input_fn

    saved_dir = estimator.export_savedmodel(
        export_dir,
        serving_input_fn,
        assets_extra=None,
        as_text=False,
        checkpoint_path=init_checkpoint,
        strip_default_attrs=False)

    model_name = FLAGS.trtis_model_name

    model_folder = export_dir + "/trtis_models/" + model_name
    version_folder = model_folder + "/" + str(FLAGS.trtis_model_version)
    final_model_folder = version_folder + "/model.savedmodel"

    if not os.path.exists(version_folder):
        os.makedirs(version_folder)

    if (not os.path.exists(final_model_folder)):
        os.rename(saved_dir, final_model_folder)
        print("Model saved to dir", final_model_folder)
    else:
        if (FLAGS.trtis_model_overwrite):
            shutil.rmtree(final_model_folder)
            os.rename(saved_dir, final_model_folder)
            print("WARNING: Existing model was overwritten. Model dir: {}".format(final_model_folder))
        else:
            print("ERROR: Could not save TRTIS model. Folder already exists. Use '--trtis_model_overwrite=True' if you would like to overwrite an existing model. Model dir: {}".format(final_model_folder))
            return

    # Now build the config for TRTIS. Check to make sure we can overwrite it, if it exists
    config_filename = os.path.join(model_folder, "config.pbtxt")

    if (os.path.exists(config_filename) and not FLAGS.trtis_model_overwrite):
        print("ERROR: Could not save TRTIS model config. Config file already exists. Use '--trtis_model_overwrite=True' if you would like to overwrite an existing model config. Model config: {}".format(config_filename))
        return

    config_template = r"""
name: "{model_name}"
platform: "tensorflow_savedmodel"
max_batch_size: {max_batch_size}
input [
    {{
        name: "unique_ids"
        data_type: TYPE_INT32
        dims: [ 1 ]
        reshape: {{ shape: [ ] }}
    }},
    {{
        name: "segment_ids"
        data_type: TYPE_INT32
        dims: {seq_length}
    }},
    {{
        name: "input_ids"
        data_type: TYPE_INT32
        dims: {seq_length}
    }},
    {{
        name: "input_mask"
        data_type: TYPE_INT32
        dims: {seq_length}
    }}
    ]
    output [
    {{
        name: "end_logits"
        data_type: TYPE_FP32
        dims: {seq_length}
    }},
    {{
        name: "start_logits"
        data_type: TYPE_FP32
        dims: {seq_length}
    }}
]
{dynamic_batching}
instance_group [
    {{
        count: {engine_count}
        kind: KIND_GPU
        gpus: [{gpu_list}]
    }}
]"""

    batching_str = ""
    max_batch_size = FLAGS.trtis_max_batch_size

    if (FLAGS.trtis_dyn_batching_delay > 0):

        # Use only full and half full batches
        pref_batch_size = [int(max_batch_size / 2.0), max_batch_size]

        batching_str = r"""
dynamic_batching {{
    preferred_batch_size: [{0}]
    max_queue_delay_microseconds: {1}
}}""".format(", ".join([str(x) for x in pref_batch_size]), int(FLAGS.trtis_dyn_batching_delay * 1000.0))

    config_values = {
        "model_name": model_name,
        "max_batch_size": max_batch_size,
        "seq_length": FLAGS.max_seq_length,
        "dynamic_batching": batching_str,
        "gpu_list": ", ".join([x.name.split(":")[-1] for x in device_lib.list_local_devices() if x.device_type == "GPU"]),
        "engine_count": FLAGS.trtis_engine_count
    }

    with open(model_folder + "/config.pbtxt", "w") as file:

        final_config_str = config_template.format_map(config_values)
        file.write(final_config_str)

def main(_):
  os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_lazy_compilation=false" #causes memory fragmentation for bert leading to OOM
  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.horovod:
    hvd.init()
  if FLAGS.use_fp16:
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"

  bert_config = modeling_func.BertConfig.from_json_file(FLAGS.bert_config_file)

  if bert_config.use_rel_pos_embeddings:
    tf.logging.info("Use relative position embeddings")
    tf.logging.info("max_rel_positions %d" % bert_config.max_rel_positions)

  if bert_config.roberta:
    tf.logging.info("Load Roberta for fine-tuning!")

  if bert_config.electra:
    tf.logging.info("Load Electra for fine-tuning!")


  validate_flags_or_throw(bert_config)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  master_process = True
  training_hooks = []
  global_batch_size = FLAGS.train_batch_size * FLAGS.num_accumulation_steps
  hvd_rank = 0

  config = tf.ConfigProto()
  learning_rate = FLAGS.learning_rate
  if FLAGS.horovod:

      tf.logging.info("Multi-GPU training with TF Horovod")
      tf.logging.info("hvd.size() = %d hvd.rank() = %d", hvd.size(), hvd.rank())
      global_batch_size = FLAGS.train_batch_size * hvd.size() * FLAGS.num_accumulation_steps
      # learning_rate = learning_rate * hvd.size()
      master_process = (hvd.rank() == 0)
      hvd_rank = hvd.rank()
      config.gpu_options.visible_device_list = str(hvd.local_rank())
      if hvd.size() > 1:
          training_hooks.append(hvd.BroadcastGlobalVariablesHook(0))
  if FLAGS.use_xla:
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    config.graph_options.rewrite_options.memory_optimization = rewriter_config_pb2.RewriterConfig.NO_MEM_OPT
    if FLAGS.use_fp16:
      tf.enable_resource_variables()

  run_config = tf.estimator.RunConfig(
      model_dir=FLAGS.output_dir if master_process else None,
      session_config=config,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps if master_process else None,
      keep_checkpoint_max=1)

  if master_process:
      tf.logging.info("***** Configuaration *****")
      for key in FLAGS.__flags.keys():
          tf.logging.info('  {}: {}'.format(key, getattr(FLAGS, key)))
      tf.logging.info("**************************")

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  training_hooks.append(LogTrainRunHook(global_batch_size, hvd_rank, FLAGS.save_checkpoints_steps))

  # Prepare Training Data
  if FLAGS.do_train:
    train_examples = read_squad_examples(
        input_file=FLAGS.train_file, is_training=True,
        version_2_with_negative=FLAGS.version_2_with_negative)
    num_train_steps = int(
        len(train_examples) / global_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    # Pre-shuffle the input to avoid having to make a very large shuffle
    # buffer in in the `input_fn`.
    rng = random.Random(12345)
    rng.shuffle(train_examples)

    start_index = 0
    end_index = len(train_examples)
    tmp_filenames = [os.path.join(FLAGS.output_dir, "train.tf_record")]

    if FLAGS.horovod:
      tmp_filenames = [os.path.join(FLAGS.output_dir, "train.tf_record{}".format(i)) for i in range(hvd.size())]
      num_examples_per_rank = len(train_examples) // hvd.size()
      remainder = len(train_examples) % hvd.size()
      if hvd.rank() < remainder:
        start_index = hvd.rank() * (num_examples_per_rank+1)
        end_index = start_index + num_examples_per_rank + 1
      else:
        start_index = hvd.rank() * num_examples_per_rank + remainder
        end_index = start_index + (num_examples_per_rank)


  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      hvd=None if not FLAGS.horovod else hvd,
      use_fp16=FLAGS.use_fp16,
      use_one_hot_embeddings=False,
  )

  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config)

  if FLAGS.do_train:

    # We write to a temporary file to avoid storing very large constant tensors
    # in memory.
    train_writer = FeatureWriter(
        filename=tmp_filenames[hvd_rank],
        is_training=True)
    convert_examples_to_features(
        examples=train_examples[start_index:end_index],
        tokenizer=tokenizer,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        is_training=True,
        output_fn=train_writer.process_feature,
        verbose_logging=FLAGS.verbose_logging)
    train_writer.close()

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num orig examples = %d", end_index - start_index)
    tf.logging.info("  Num split examples = %d", train_writer.num_features)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    tf.logging.info("  LR = %f", learning_rate)
    del train_examples

    train_input_fn = input_fn_builder(
        input_file=tmp_filenames,
        batch_size=FLAGS.train_batch_size,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True,
        hvd=None if not FLAGS.horovod else hvd)

    train_start_time = time.time()
    estimator.train(input_fn=train_input_fn, hooks=training_hooks, max_steps=num_train_steps)
    train_time_elapsed = time.time() - train_start_time
    train_time_wo_overhead = training_hooks[-1].total_time
    avg_sentences_per_second = num_train_steps * global_batch_size * 1.0 / train_time_elapsed
    ss_sentences_per_second = (num_train_steps - training_hooks[-1].skipped) * global_batch_size * 1.0 / train_time_wo_overhead

    if master_process:
        tf.logging.info("-----------------------------")
        tf.logging.info("Total Training Time = %0.2f for Sentences = %d", train_time_elapsed,
                        num_train_steps * global_batch_size)
        tf.logging.info("Total Training Time W/O Overhead = %0.2f for Sentences = %d", train_time_wo_overhead,
                        (num_train_steps - training_hooks[-1].skipped) * global_batch_size)
        tf.logging.info("Throughput Average (sentences/sec) with overhead = %0.2f", avg_sentences_per_second)
        tf.logging.info("Throughput Average (sentences/sec) = %0.2f", ss_sentences_per_second)
        tf.logging.info("-----------------------------")


  if FLAGS.export_trtis and master_process:
    export_model(estimator, FLAGS.output_dir, FLAGS.init_checkpoint)

  if FLAGS.do_predict and master_process:
    eval_examples = read_squad_examples(
        input_file=FLAGS.predict_file, is_training=False,
        version_2_with_negative=FLAGS.version_2_with_negative)

    # Perform evaluation on subset, useful for profiling
    if FLAGS.num_eval_iterations is not None:
        eval_examples = eval_examples[:FLAGS.num_eval_iterations*FLAGS.predict_batch_size]

    eval_writer = FeatureWriter(
        filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
        is_training=False)
    eval_features = []

    def append_feature(feature):
      eval_features.append(feature)
      eval_writer.process_feature(feature)

    convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        is_training=False,
        output_fn=append_feature,
        verbose_logging=FLAGS.verbose_logging)
    eval_writer.close()

    tf.logging.info("***** Running predictions *****")
    tf.logging.info("  Num orig examples = %d", len(eval_examples))
    tf.logging.info("  Num split examples = %d", len(eval_features))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_input_fn = input_fn_builder(
        input_file=eval_writer.filename,
        batch_size=FLAGS.predict_batch_size,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

    all_results = []
    eval_hooks = [LogEvalRunHook(FLAGS.predict_batch_size)]
    eval_start_time = time.time()
    for result in estimator.predict(
        predict_input_fn, yield_single_examples=True, hooks=eval_hooks):
      if len(all_results) % 1000 == 0:
        tf.logging.info("Processing example: %d" % (len(all_results)))
      unique_id = int(result["unique_ids"])
      start_logits = [float(x) for x in result["start_logits"].flat]
      end_logits = [float(x) for x in result["end_logits"].flat]
      all_results.append(
          RawResult(
              unique_id=unique_id,
              start_logits=start_logits,
              end_logits=end_logits))

    eval_time_elapsed = time.time() - eval_start_time
    eval_time_wo_overhead = eval_hooks[-1].total_time

    time_list = eval_hooks[-1].time_list
    time_list.sort()
    num_sentences = (eval_hooks[-1].count - eval_hooks[-1].skipped) * FLAGS.predict_batch_size

    avg = np.mean(time_list)
    cf_50 = max(time_list[:int(len(time_list) * 0.50)])
    cf_90 = max(time_list[:int(len(time_list) * 0.90)])
    cf_95 = max(time_list[:int(len(time_list) * 0.95)])
    cf_99 = max(time_list[:int(len(time_list) * 0.99)])
    cf_100 = max(time_list[:int(len(time_list) * 1)])
    ss_sentences_per_second = num_sentences * 1.0 / eval_time_wo_overhead

    tf.logging.info("-----------------------------")
    tf.logging.info("Total Inference Time = %0.2f for Sentences = %d", eval_time_elapsed,
                    eval_hooks[-1].count * FLAGS.predict_batch_size)
    tf.logging.info("Total Inference Time W/O Overhead = %0.2f for Sentences = %d", eval_time_wo_overhead,
                    (eval_hooks[-1].count - eval_hooks[-1].skipped) * FLAGS.predict_batch_size)
    tf.logging.info("Summary Inference Statistics")
    tf.logging.info("Batch size = %d", FLAGS.predict_batch_size)
    tf.logging.info("Sequence Length = %d", FLAGS.max_seq_length)
    tf.logging.info("Precision = %s", "fp16" if FLAGS.use_fp16 else "fp32")
    tf.logging.info("Latency Confidence Level 50 (ms) = %0.2f", cf_50 * 1000)
    tf.logging.info("Latency Confidence Level 90 (ms) = %0.2f", cf_90 * 1000)
    tf.logging.info("Latency Confidence Level 95 (ms) = %0.2f", cf_95 * 1000)
    tf.logging.info("Latency Confidence Level 99 (ms) = %0.2f", cf_99 * 1000)
    tf.logging.info("Latency Confidence Level 100 (ms) = %0.2f", cf_100 * 1000)
    tf.logging.info("Latency Average (ms) = %0.2f", avg * 1000)
    tf.logging.info("Throughput Average (sentences/sec) = %0.2f", ss_sentences_per_second)
    tf.logging.info("-----------------------------")

    output_prediction_file = os.path.join(FLAGS.output_dir, "predictions.json")
    output_nbest_file = os.path.join(FLAGS.output_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(FLAGS.output_dir, "null_odds.json")

    write_predictions(eval_examples, eval_features, all_results,
                      FLAGS.n_best_size, FLAGS.max_answer_length,
                      FLAGS.do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file)


if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
