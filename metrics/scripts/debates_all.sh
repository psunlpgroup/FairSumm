for id in 2 3 4;
do
  for score in n-gram bart bert;
  do
    data_path=/home/yfz5488/fairsumm/preprocessing/preprocessed_datasets/oxforddebates.json
    pred_path=/home/yfz5488/fairsumm/models/GPTs/results/OxfordDebates/predictions_${id}.txt
    save_path=results/OxfordDebates/exp${id}_${score}.json
    attribute=speaker
    name=oxforddebates_exp${id}
    score=${score}
    device=4

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