#!/usr/bin/env python
"""Data helper function for BioASQ."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import json
import math
import os
import random
import tokenization
import itertools
import tensorflow as tf


class BioASQExample(object):
  """A single training/test example for simple sequence classification.

     For examples without an answer, labels is -1.
  """

  def __init__(self,
               qas_id,
               qid,
               question_text,
               doc_tokens,
               boolean_label=None):
    self.qas_id = qas_id
    self.qid = qid
    self.question_text = question_text
    self.doc_tokens = doc_tokens
    self.boolean_label = boolean_label

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
    s += ", question_text: %s" % (
        tokenization.printable_text(self.question_text))
    s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
    if self.boolean_label:
        s += ", boolean_label: [%s]" % self.boolean_label
    return s

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               doc_span_index,
               tokens,
               input_ids,
               input_mask,
               segment_ids,
               start_position=None,
               end_position=None):
    self.unique_id = unique_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.start_position = start_position
    self.end_position = end_position


class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, is_training):
    self.filename = filename
    self.is_training = is_training
    self.num_features = 0
    self._writer = tf.python_io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)

    if self.is_training:
      features["start_positions"] = create_int_feature([feature.start_position])
      features["end_positions"] = create_int_feature([feature.end_position])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


def convert_grouped_examples(qid, q_group):
    """Converts a group of examples into one example."""
    list_doc_tokens = []
    for one_example in list(q_group):
      list_doc_tokens.append(one_example.doc_tokens)

    merged_doc_tokens = sum(list_doc_tokens, [])
    return BioASQExample(
        qas_id="{0}_0".format(qid),
        qid=qid,
        question_text=one_example.question_text,
        doc_tokens=merged_doc_tokens,
        boolean_label=one_example.boolean_label,
    )


def process_single_line_bioasq(line, delimiter="\t"):
    """Processes a single example of BioASQ, a pair of question and snippet."""
    # For each line, the order items are:
    # [0] id: {question_id}_{numbering},
    # [1] question: String for the question,
    # [2] snippet: String for the snippet text sentence,
    # [3] label: {Yes, No} answer label.
    items = line.strip().split(delimiter)

    return BioASQExample(
        qas_id=items[0],
        qid=items[0].split("_")[0],
        question_text=items[1],
        doc_tokens=items[2].split(),
        boolean_label=items[3],
    )


def read_bioasq_examples(input_file, group_by_question=False, delimiter="\t"):
    """Reads in bioasq examples."""
    with open(input_file, "r") as fin:
        examples = [process_single_line_bioasq(line, delimiter=delimiter) for line in fin]

    if group_by_question:
        key_func = lambda x: str(x.qid)

        examples = [
            convert_grouped_examples(qid, q_group)
            for qid, q_group in itertools.groupby(
                    sorted(examples, key=key_func), key=key_func)
        ]

    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 output_fn, verbose_logging=False,
                                 cls_token="[CLS]", sep_token="[SEP]",
                                 pad_token_id=0, yes_token="yes",
                                 no_token="no"):
  """Loads a data file into a list of `InputBatch`s."""

  unique_id = 1000000000

  for (example_index, example) in enumerate(examples):
    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > max_query_length:
      query_tokens = query_tokens[0:max_query_length]

    all_doc_tokens = tokenizer.tokenize(" ".join(example.doc_tokens))

    # The -6 accounts for [CLS], yes, no, [SEP] [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 6

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
      length = len(all_doc_tokens) - start_offset
      if length > max_tokens_for_doc:
        length = max_tokens_for_doc
      doc_spans.append(_DocSpan(start=start_offset, length=length))
      if start_offset + length == len(all_doc_tokens):
        break
      start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
      tokens = []
      segment_ids = []

      # Prefixes [CLS].
      tokens.append(cls_token)
      segment_ids.append(0)

      # Addes yes (position 1) and no (position 2) predictions.
      tokens.append(yes_token)
      segment_ids.append(0)
      tokens.append(no_token)
      segment_ids.append(0)

      # Separates the labels with question and body text.
      tokens.append(sep_token)
      segment_ids.append(0)

      for token in query_tokens:
        tokens.append(token)
        segment_ids.append(0)

      tokens.append(sep_token)
      segment_ids.append(0)

      for i in range(doc_span.length):
        split_token_index = doc_span.start + i
        tokens.append(all_doc_tokens[split_token_index])
        segment_ids.append(1)

      tokens.append(sep_token)
      segment_ids.append(1)

      input_ids = tokenizer.convert_tokens_to_ids(tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      while len(input_ids) < max_seq_length:
        input_ids.append(pad_token_id)
        input_mask.append(0)
        segment_ids.append(0)

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length

      start_position = None
      end_position = None

      if example.boolean_label:
          if example.boolean_label == "yes":
              start_position = 1
              end_position = 1
          elif example.boolean_label == "no":
              start_position = 2
              end_position = 2
          else:
              raise ValueError(
                  "Unknown boolean_label: %s" % example.boolean_label)

      if verbose_logging and example_index < 20:
        tf.logging.info("*** Example ***")
        tf.logging.info("unique_id: %s" % (unique_id))
        tf.logging.info("example_index: %s" % (example_index))
        tf.logging.info("doc_span_index: %s" % (doc_span_index))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("token_is_max_context: %s" % " ".join([
            "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
        ]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info(
            "input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info(
            "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        if is_training and not example.is_impossible:
          answer_text = " ".join(tokens[start_position:(end_position + 1)])
          tf.logging.info("start_position: %d" % (start_position))
          tf.logging.info("end_position: %d" % (end_position))
          tf.logging.info(
              "answer: %s" % (tokenization.printable_text(answer_text)))

      feature = InputFeatures(
          unique_id=unique_id,
          example_index=example_index,
          doc_span_index=doc_span_index,
          tokens=tokens,
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids,
          start_position=start_position,
          end_position=end_position,
      )

      # Run callback
      output_fn(feature)

      unique_id += 1
