# Posterior Differential Regularization with f-divergence for ImprovingModel Robustness

This repository includes the codes for the paper
[Posterior Differential Regularization with f-divergence for ImprovingModel Robustness](https://arxiv.org/abs/2010.12638).

This code base will be based on Tensorflow.
If you are interested in a PyTorch implementation, please see [MTDNN](https://github.com/namisan/mt-dnn).

```
@inproceedings{cheng-etal-2021-posterior,
    title = "Posterior Differential Regularization with f-divergence for Improving Model Robustness",
    author = "Cheng, Hao  and
      Liu, Xiaodong  and
      Pereira, Lis  and
      Yu, Yaoliang  and
      Gao, Jianfeng",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.85",
    doi = "10.18653/v1/2021.naacl-main.85",
    pages = "1078--1089",
}
```

## Requirements
* Python >= 3.6
* Tensorflow 1.14

The code in this repo has been tested with Tensorflow 1.14 on V100-32GB.
We highly recommend using the docker file for creating the enviroment.
All the following sample commands are based on using docker.

Build the docker image:
```
cd docker_file
docker build -t united_qa:v1.0 -f Dockerfile.cuda11 .
```

All the following commands are run within the docker enviroment.

## Train w/ PDR for MNLI
First, download the MNLI data into a {glue_dir} folder.
Then, the following command train the MNLI model w/ Hellinger loss in VAT fashion.
If you only want to do double forward, just set {vat_reg_rate=0.0} and {double_forward_reg_rate} to set a non-zero value.
```
python run_classifier.py \
  --task_name=MNLI \
  --do_train=true \
  --do_eval=true \
  --data_dir=${glue_dir} \
  --vocab_file=${vocab} \
  --bert_config_file=${model_config} \
  --init_checkpoint=${init_ckpt} \
  --max_seq_length=${max_seq_length} \
  --doc_stride=${doc_stride} \
  --train_batch_size=${batch_size} \
  --eval_batch_size=${eval_batch_size} \
  --learning_rate=$learning_rate \
  --num_train_epochs=$num_train_epochs \
  --vat_reg_rate=1.0 \
  --double_forward_reg_rate=0.0 \
  --double_forward_loss="hellinger" \
  --noise_epsilon=1e-3 \
  --noise_normalizer="L2" \
  --noise_type="tok" \
  --output_dir=$output_dir |& tee ${output_dir}/log.log
```

## Train w/ PDR for SQuAD
First, download the SQuADv1/v2 data into a {squad_dir} folder.

For training, different from the classification case, we need to model perturbations for the start and end positions for SQuAD (reading comprehension).
The following command train the SQuAD v2 model w/ JS loss in VAT fashion.
Similarly, if you only want to do double forward, just set {vat_reg_rate=0.0} and {double_forward_reg_rate} to set a non-zero value.
```
python run_squad.py \
  --vocab_file=${vocab_file} \
  --bert_config_file=${model_config} \
  --init_checkpoint=${model_ckpt} \
  --do_train=True \
  --train_file=$squad_dir/train-v2.0.json \
  --do_predict=True \
  --predict_file=$squad_dir/dev-v2.0.json \
  --train_batch_size=${batch_size} \
  --predict_batch_size=${batch_size} \
  --learning_rate=${learning_rate} \
  --num_train_epochs=${num_train_epochs} \
  --max_seq_length=384 \
  --max_query_length=64 \
  --vat_reg_rate=1.0 \
  --jacobian_reg_rate=0.0 \
  --double_forward_reg_rate=0.0 \
  --double_forward_loss="js" \
  --noise_epsilon=1e-3 \
  --noise_normalizer="L2" \
  --doc_stride=128 \
  --version_2_with_negative=True \
  --output_dir=${output_dir} |& tee ${output_dir}/log.log
```

Then, we can evaluate the predictions using
```
python qa_utils/evaluate-v2.0.py \
    ${squad_dir}/dev-v2.0.json \
    ${output_dir}/predictions.json |& tee ${output_dir}/dev.metrics
```
