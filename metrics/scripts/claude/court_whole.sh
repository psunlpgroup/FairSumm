data_path=/scratch1/yfz5488/fairsumm/preprocessing/preprocessed_datasets/supremecourt_whole.json
pred_path=/scratch1/yfz5488/fairsumm/models/Claude/results/postprocessed/court_whole.txt
save_path=results/SupremeCourt/claude_ngram.json
attribute=speaker
name=supremecourt_whole_claude
score=n-gram
device=1

python eval.py \
 --data_path ${data_path} \
 --pred_path ${pred_path} \
 --save_path ${save_path} \
 --attribute ${attribute} \
 --name ${name} \
 --score ${score} \
 --device ${device}
