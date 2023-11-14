data_path=/scratch1/yfz5488/fairsumm/preprocessing/preprocessed_datasets/oxforddebates_whole.json
pred_path=/scratch1/yfz5488/fairsumm/models/Claude/results/postprocessed/debates_whole.txt
save_path=results/OxfordDebates/claude_ngram.json
attribute=speaker
name=oxforddebates_whole_claude
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
