"""Extracting predictions from n-best file with multiple paragrpahs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import codecs
import json
import pdb
from absl import app
import tensorflow as tf
import six
import numpy as np


flags = tf.flags

FLAGS = flags.FLAGS
flags.DEFINE_float(
    "decay", 4.0,
    "A decay power used to downweight answers in lower ranked passages.")
flags.DEFINE_bool(
    "sum", True,
    "Whether to use the sum or max aggregation.")
flags.DEFINE_integer(
    "max_para", 8,
    "The lowest ranked paragraph to use -- ranks are 0 to m.")
flags.DEFINE_string("nbest_file", "",
                    "A file with nbest predictions.")
flags.DEFINE_string("predictions_file", "",
                    "A file to write final predictions to.")
flags.DEFINE_bool("convert", False,
                  "Whether to convert predictions to lines.")
flags.DEFINE_bool("use_rank", True,
                  "Whether to use the ranking information, only sum=True")
flags.DEFINE_bool("use_doc_score", False,
                  "Whether to use the doc score information.")



def extract_strings(json_file_in, guess_file_out):
    """Extracts the answer string from the file."""
    result_lines = []
    with tf.gfile.Open(json_file_in, "r") as reader:
        input_data = json.load(reader, object_pairs_hook=collections.OrderedDict)
        for eid, predictions in six.iteritems(input_data):
            print(eid)
    result_lines.append(predictions)
    with tf.gfile.Open(guess_file_out, mode="w") as fout:
        for l in result_lines:
            fout.write(l.encode("utf-8")+"\n")


def make_predictions(nbest_file, predictions_file):
    # for each qid we record a map from answer string to occurences
    # an occurrence has a probability and a paragraph index
    RawOccurrence = collections.namedtuple("RawOccurrence", ["prob", "rank"])
    qid_to_answers = {}
    qid_to_aggregated_answers = {}
    # intergate the rank as prob * rank**(-1.0/4)
    result_dict = {}

    use_rank = FLAGS.use_rank if FLAGS.sum else False
    if use_rank:
        tf.logging.info("Using rank to re-scale predictions!")
    else:
        tf.logging.info("Using raw predictions!")

    with open(nbest_file) as reader:
        input_data = json.load(reader)
        for qid_para, predictions in six.iteritems(input_data):
            qid, rank = qid_para.split("-")
            rank = int(rank)
            if rank >= FLAGS.max_para:
                continue
            if qid not in qid_to_answers:
                qid_to_answers[qid] = collections.defaultdict(list)
                qid_to_aggregated_answers[qid] = collections.defaultdict(float)

            for p in predictions:
                ans_string = p["text"]
                if FLAGS.use_doc_score:
                    prob = np.exp(p["start_logit"] + p["end_logit"] - p["doc_score"])
                    assert prob > -1e-3
                else:
                    prob = p["probability"]
                qid_to_answers[qid][ans_string].append(
                    RawOccurrence(prob=prob, rank=rank)
                    )
                if FLAGS.use_rank:
                    score = prob*((rank+1)**(
                        -1.0/FLAGS.decay
                        ))
                else:
                    score = prob

                if FLAGS.sum:
                    qid_to_aggregated_answers[qid][ans_string] += score
                else:
                  if qid_to_aggregated_answers[qid][ans_string] < score:
                      qid_to_aggregated_answers[qid][ans_string] = score

    # now choose the best guess for each qid
    for qid, answer_dict in six.iteritems(qid_to_aggregated_answers):
        q_ans_ordered = collections.OrderedDict(sorted(answer_dict.items(),
                                                       key=lambda t: -t[1]))
        for ans, _ in six.iteritems(q_ans_ordered):
            if ans:
                result_dict[qid] = ans
                break

    with open(predictions_file, mode="wt") as fout:
        fout.write(json.dumps(result_dict,
                              sort_keys=True,
                              indent=4,
                              ensure_ascii=False))


def main(_):
    if FLAGS.convert:
        extract_strings(FLAGS.predictions_file, FLAGS.predictions_file+".txt")
        return
    make_predictions(FLAGS.nbest_file, FLAGS.predictions_file)


if __name__ == '__main__':
    app.run(main)
