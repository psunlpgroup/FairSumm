for score in n-gram bart bert;
do
  data_path=/home/yfz5488/fairsumm/preprocessing/preprocessed_datasets/yelp.json
  pred_path=/home/yfz5488/fairsumm/models/Alpaca/results/postprocessed/yelp.json
  save_path=results/Yelp/alpaca_${score}.json
  attribute=sentiment
  name=yelp_alpaca
  score=${score}
  device=2

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