for score in n-gram bart bert;
do
  data_path=/home/yfz5488/fairsumm/preprocessing/preprocessed_datasets/oxforddebates.json
  pred_path=/home/yfz5488/fairsumm/models/Alpaca/results/postprocessed/oxforddebates.json
  save_path=results/OxfordDebates/alpaca_${score}.json
  attribute=speaker
  name=oxforddebates_alpaca
  score=${score}
  device=4

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