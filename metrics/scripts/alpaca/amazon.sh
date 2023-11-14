for score in n-gram bart bert;
do
  data_path=/scratch1/yfz5488/fairsumm/preprocessing/preprocessed_datasets/amazon.json
  pred_path=/scratch1/yfz5488/fairsumm/models/Alpaca/results/postprocessed/amazon.json
  save_path=results/Amazon/alpaca_${score}.json
  attribute=rating
  name=amazon_alpaca
  score=${score}
  device=1

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