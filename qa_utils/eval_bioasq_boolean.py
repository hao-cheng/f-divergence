#!/usr/bin/env python
"""Evaluation script for BioASQ boolean case."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import os
import sys
import argparse
import json
import itertools
import collections
import numpy as np


def compute_metrics_aggregate(prediction_fn, label_fn, delimiter="\t"):
    """Reads predictions."""
    with open(prediction_fn, "r") as fin:
        predictions = json.load(fin)

    results = {}
    qid_to_label = {}
    qid_to_predict = collections.defaultdict(lambda: collections.defaultdict(float))
    num_correct, num_count = 0, 0
    with open(label_fn, "r") as fin:
        for line in fin:
            items = line.strip().split(delimiter)
            qas_id = items[0].strip()
            qid = qas_id.split("_")[0]
            if qas_id not in predictions:
                # This can happen when training with merged example.
                qas_id = "{0}_0".format(qid)
            pred_dict = predictions[qas_id][0]
            predict = pred_dict["text"]
            if items[3] == predict:
                num_correct += 1
            num_count += 1

            qid_to_predict[qid][predict] += pred_dict["probability"]
            qid_to_label[qid] = items[3]

    accuracy = float(num_correct) / num_count
    print("Number of examples: %d" % num_count)
    print("Instance-level accuracy: %.2f" % (accuracy * 100))
    results["instance_accuracy"] = accuracy * 100
    results["num_instance"] = num_count

    doc_correct, num_doc = 0, 0
    for qid, preds in qid_to_predict.items():
        predict = "yes" if preds["yes"] > preds["no"] else "no"
        if predict == qid_to_label[qid]:
            doc_correct += 1

        num_doc += 1

    doc_accuracy = float(doc_correct) / num_doc
    print("Number of questions: %d" % num_doc)
    print("Question-level accuracy: %.2f" % (doc_accuracy * 100))
    results["question_accuracy"] = doc_accuracy * 100
    results["num_question"] = num_doc

    return results


def compute_metrics(prediction_fn, label_fn, delimiter="\t"):
    """Reads predictions."""
    with open(prediction_fn, "r") as fin:
        predictions = json.load(fin)

    results = {}
    joint_results = []
    num_correct, num_count = 0, 0
    with open(label_fn, "r") as fin:
        for line in fin:
            items = line.strip().split(delimiter)
            qas_id = items[0].strip()
            qid = qas_id.split("_")[0]
            joint_results.append({
                "qas_id": qas_id,
                "qid": qid,
                "label": items[3],
                "prediction": predictions[qas_id],
            })
            if items[3] == predictions[qas_id]:
                num_correct += 1
            num_count += 1

    accuracy = float(num_correct) / num_count
    print("Number of examples: %d" % num_count)
    print("Instance-level accuracy: %.2f" % (accuracy * 100))
    results["instance_accuracy"] = accuracy * 100
    results["num_instance"] = num_count

    key_func = lambda x: str(x["qid"])
    doc_correct, num_doc = 0, 0
    for qid, group in itertools.groupby(
            sorted(joint_results, key=key_func), key=key_func):
        q_results = list(group)
        votes = dict(collections.Counter([item["prediction"] for item in q_results]))
        if votes.get("yes", 0) > votes.get("no", 0):
            doc_pred = "yes"
        else:
            doc_pred = "no"

        if doc_pred == q_results[0]["label"]:
            doc_correct += 1

        num_doc += 1

    doc_accuracy = float(doc_correct) / num_doc
    print("Number of questions: %d" % num_doc)
    print("Question-level accuracy: %.2f" % (doc_accuracy * 100))
    results["question_accuracy"] = doc_accuracy * 100
    results["num_question"] = num_doc

    return results


def main():
    cmdline_parser = argparse.ArgumentParser(
        description=__doc__
    )
    cmdline_parser.add_argument(
        'labels',
        help='label file',
    )
    cmdline_parser.add_argument(
        'predictions',
        help='prediction file',
    )
    cmdline_parser.add_argument(
        'metric_filename',
        help='metric file to dump',
    )

    args = cmdline_parser.parse_args()

    metrics = compute_metrics_aggregate(args.predictions, args.labels)
    print(metrics)

    with open(args.metric_filename, "wt") as fout:
        json.dump(metrics, fout)


if __name__ == '__main__':
    main()

