#! /bin/bash

key=
dataset=OxfordDebates
exp_name=4
data_folder="/home/yfz5488/fairsumm/datasets/Oxford-style debates"
max_token=512
python run_gpt4.py \
  --openai_key ${key} \
  --dataset ${dataset} \
  --data_folder "/home/yfz5488/fairsumm/datasets/Oxford-style debates" \
  --exp_name ${exp_name} \
  --max_token ${max_token}
