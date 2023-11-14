#for id in 2 3;
for id in 4;
do
  for score in n-gram bart bert;
  do
    data_path=/scratch1/yfz5488/fairsumm/preprocessing/preprocessed_datasets/supremecourt.json
    pred_path=/scratch1/yfz5488/fairsumm/models/GPTs/results/SupremeCourt/predictions_${id}.txt
    save_path=results/SupremeCourt/exp${id}_${score}.json
    attribute=speaker
    name=supremecourt_exp${id}
    score=${score}
    device=3

    python eval.py \
     --data_path ${data_path} \
     --pred_path ${pred_path} \
     --save_path ${save_path} \
     --attribute ${attribute} \
     --name ${name} \
     --score ${score} \
     --device ${device}
  done
done