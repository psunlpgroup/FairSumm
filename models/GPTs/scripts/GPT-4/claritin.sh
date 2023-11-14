#! /bin/bash

key=
dataset=Claritin
exp_name=18
data_folder=/home/yfz5488/fairsumm/processed_datasets/claritin.json
max_token=512
python run_gpt4.py \
  --openai_key ${key} \
  --dataset ${dataset} \
  --data_folder ${data_folder} \
  --exp_name ${exp_name} \
  --max_token ${max_token}