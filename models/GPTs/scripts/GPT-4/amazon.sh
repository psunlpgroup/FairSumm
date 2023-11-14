#! /bin/bash

key=
dataset=Fewsum_origin
subdataset=amazon
exp_name=4
data_folder=/mnt/backups/yfz5488/fairsumm/preprocessing/preprocessed_datasets/amazon.json
max_token=512
python run_gpt4.py \
  --openai_key ${key} \
  --dataset ${dataset} \
  --subdataset ${subdataset} \
  --data_folder ${data_folder} \
  --exp_name ${exp_name} \
  --max_token ${max_token}