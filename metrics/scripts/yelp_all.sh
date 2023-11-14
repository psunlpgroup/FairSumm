for id in 10; # 1 2 4;
do
  for score in n-gram bart bert;
  do
    data_path=/scratch1/yfz5488/fairsumm/preprocessing/preprocessed_datasets/yelp.json
    pred_path=/scratch1/yfz5488/fairsumm/models/GPTs/results/Fewsum_origin/yelp/predictions_${id}.txt
    save_path=results/Yelp/exp${id}_${score}.json
    attribute=sentiment
    name=yelp_exp${id}
    score=${score}
    device=2

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