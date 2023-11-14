# for id in 1 2 3;
for id in 10;
do
  for score in n-gram bart bert;
  do
    data_path=/scratch1/yfz5488/fairsumm/preprocessing/preprocessed_datasets/election.json
    pred_path=/scratch1/yfz5488/fairsumm/models/GPTs/results/Election/predictions_${id}.txt
    save_path=results/Election/exp${id}_${score}.json
    attribute=politics
    name=election_exp${id}
    score=${score}
    device=0

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