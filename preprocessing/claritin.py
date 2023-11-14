import os.path

from dataset_loader import Claritin
import json
gender_list = ['male','female']
loader = Claritin.Loader()
dataset_path="/home/yfz5488/fairsumm/datasets/FairSumm-master/Dataset/Claritin/"
# def load(self, dataset_path, num_sample=100, num_tweets=50, male_ratio=0.5, min_length=10, max_length=70, shuffle=True):
dataset = []
for num_tweets in [10, 30, 50]:
    for male_ratio in [0.2,0.5,0.8]:
        loader = Claritin.Loader()
        data, _, value_inputs = loader.load(dataset_path=dataset_path, num_sample=150, num_tweets=num_tweets, male_ratio=male_ratio, shuffle=True, verbose=True)
        data = [x[0] for x in data['test']]

        for sample, values in zip(data, value_inputs):
            # Get units
            value_as_input = [" || ".join(x) for x in values]
            units = []
            for i in range(2):
                for review in values[i]:
                    units.append({"text":review,"value":{"gender":gender_list[i]}})

            # Append to dataset
            dataset.append({
                "model_input": sample,
                "value_input": {'gender': value_as_input},
                "value_mapping": {'gender': gender_list},
                "meta":{"num_tweets": num_tweets,
                        "male_ratio": male_ratio,
                        "type": "tweets",
                        "seperator": " || "},
                "units": units
            })

# Save to file
if not os.path.exists("preprocessed_datasets"):
    os.makedirs("preprocessed_datasets")

print("Total samples:", len(dataset))
json.dump(dataset, open("preprocessed_datasets/claritin.json",'w'), indent=2, ensure_ascii=False)