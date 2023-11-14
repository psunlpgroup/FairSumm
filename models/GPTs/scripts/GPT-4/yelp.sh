#! /bin/bash

key=
dataset=Fewsum_origin
subdataset=yelp
exp_name=4
data_folder=/home/yfz5488/fairsumm/processed_datasets/yelp.json
max_token=512
python run_gpt4.py \
  --openai_key ${key} \
  --dataset ${dataset} \
  --subdataset ${subdataset} \
  --data_folder ${data_folder} \
  --exp_name ${exp_name} \
  --max_token ${max_token}