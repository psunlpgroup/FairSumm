for score in n-gram bart bert;
do
  data_path=/scratch1/yfz5488/fairsumm/preprocessing/preprocessed_datasets/supremecourt.json
  pred_path=/scratch1/yfz5488/fairsumm/models/Alpaca/results/postprocessed/supremecourt.json
  save_path=results/SupremeCourt/alpaca_${score}.json
  attribute=speaker
  name=supremecourt_alpaca
  score=${score}
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