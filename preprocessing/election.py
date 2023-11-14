from dataset_loader import Election
import json

dataset_path="/home/yfz5488/fairsumm/datasets/FairSumm-master/Dataset/US-Election/"
politics_list = ['repu','demo','neutral']

dataset = []
for num_tweets in [10, 30, 50]:
    for repu_ratio, demo_ratio in [(0.2,0.6),(0.4,0.4),(0.6,0.2)]:
        loader = Election.Loader()
        data_mix, _, data_value = loader.load(dataset_path=dataset_path,
                           num_sample=150, num_tweets=num_tweets,
                           demo_ratio=demo_ratio, repu_ratio=repu_ratio,
                           shuffle=True, verbose=True)
        data_mix = [x[0] for x in data_mix['test']]
        for sample, values in zip(data_mix, data_value['test']):
            # Get units
            value_as_input = [" || ".join(x) for x in values]
            units = []
            for i in range(3):
                for review in values[i]:
                    units.append({"text": review, "value": {"politics": politics_list[i]}})

            dataset.append({
                "model_input": sample,
                "value_input": {'politics': value_as_input},
                "value_mapping": {'politics': politics_list},
                "meta": {"num_tweets": num_tweets,
                         "repu_ratio": repu_ratio,
                         "demo_ratio": demo_ratio,
                         "neutral_ratio": round(1-repu_ratio-demo_ratio,1),
                         "type": "tweets",
                         "seperator": " || "},
                "units": units
            })

print("number of samples in dataset:", len(dataset))
json.dump(dataset, open("preprocessed_datasets/election.json",'w'),indent=2,ensure_ascii=False)