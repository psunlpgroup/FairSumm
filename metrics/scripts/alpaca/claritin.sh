for score in n-gram bert bart;
do
  data_path=/home/yfz5488/fairsumm/preprocessing/preprocessed_datasets/claritin.json
  pred_path=/home/yfz5488/fairsumm/models/Alpaca/results/postprocessed/claritin.json
  save_path=results/Claritin/alpaca_${score}.json
  attribute=gender
  name=claritin_alpaca
  device=3

  python eval.py \
   --data_path ${data_path} \
   --pred_path ${pred_path} \
   --save_path ${save_path} \
   --attribute ${attribute} \
   --name ${name} \
   --score ${score} \
   --device ${device} \
   --skip_empty
done