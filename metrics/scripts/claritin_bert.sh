# for id in 15 16 18 19 20 21 22 23 24 25 26 27 28;
for id in 26 27 28;
do
  data_path=/home/yfz5488/fairsumm/preprocessing/preprocessed_datasets/claritin.json
  pred_path=/home/yfz5488/fairsumm/models/GPTs/results/Claritin/predictions_${id}.txt
  save_path=results/Claritin/exp${id}_bert.json
  attribute=gender
  name=claritin_exp${id}
  score=bert
  device=1

  python eval.py \
   --data_path ${data_path} \
   --pred_path ${pred_path} \
   --save_path ${save_path} \
   --attribute ${attribute} \
   --name ${name} \
   --score ${score} \
   --device ${device}
done