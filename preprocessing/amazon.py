# Preprocess the raw amazon dataset to the format that can be run
import json
import os
import csv
import random
random.seed(42)
rating_list = [1,2,3,4,5]
TOTAL_SAMPLED = 1500
from collections import defaultdict
if __name__ == '__main__':
    folder_path = "../datasets/amazon/reviews/val"
    target_path = "preprocessed_datasets"
    data = []
    for file_name in os.listdir(folder_path):
        path = os.path.join(folder_path, file_name)
        if 'csv' not in path:
            print("not a csv file:", path)
        reader = csv.reader(open(path), delimiter='\t')
        # No need to random sample because the data itself is sampled from large data
        sample = {'units': [], 'model_input': "", 'meta':{}}
        for i, row in enumerate(reader):
            # Header: group_id	review_text	rating	category	rouge1	rouge2	rougeL	rating_dev
            if i == 0:
                continue

            sample['units'].append({
                'group_id': row[0],
                'text': row[1],
                "value": {'rating': int(float(row[2]))},
                'category': row[3],
            })

        sample['meta'] = {'id': file_name, "type": "tweets", "seperator": " || "}
        sample['model_input'] = " || ".join(x['text'] for x in sample['units'])

        value_input = defaultdict(list)
        for unit in sample['units']:
            value_input[unit["value"]["rating"]].append(unit['text'])
        value_input = [' || '.join(value_input[x]) for x in rating_list]
        sample['value_input'] = {'rating':value_input}
        sample['value_mapping'] = {'rating': rating_list }

        data.append(sample)
        if len(data) >= TOTAL_SAMPLED:
            break

    if not os.path.exists(target_path):
        os.makedirs(target_path)
    with open(os.path.join(target_path,"amazon.json"), 'w')as file:
        json.dump(data, file, indent=2, ensure_ascii=False)