for score in n-gram bert bart;
do
  data_path=/scratch1/yfz5488/fairsumm/preprocessing/preprocessed_datasets/election.json
  pred_path=/scratch1/yfz5488/fairsumm/models/Alpaca/results/postprocessed/election.json
  save_path=results/Election/alpaca_${score}.json
  attribute=politics
  name=election_alpaca
  device=0

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