#! /bin/bash

key=
dataset=Election
exp_name=3
data_folder=/mnt/backups/yfz5488/fairsumm/processed_datasets/election.json
max_token=512
python run_gpt4.py \
  --openai_key ${key} \
  --dataset ${dataset} \
  --data_folder ${data_folder} \
  --exp_name ${exp_name} \
  --max_token ${max_token}