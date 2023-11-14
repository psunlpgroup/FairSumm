# Preprocess the raw amazon dataset to the format that can be run
import json
import os
import csv
import random
random.seed(42)
rating_list = [1,2,3,4,5]

from collections import defaultdict
if __name__ == '__main__':
    folder_path = "../datasets/opinion_summ/artifacts/amazon/gold_summs"
    target_path = "preprocessed_datasets"
    data = []
    for file_name in os.listdir(folder_path):
        path = os.path.join(folder_path, file_name)
        if 'csv' not in path:
            print("not a csv file:", path)
        reader = csv.reader(open(path), delimiter='\t')
        header = []
        for i, row in enumerate(reader):
            # No need to random sample because the data itself is sampled from large data
            sample = {'units': [], 'model_input': "", 'meta': {}}
            # Header: cat	group_id	rev1	rev2	rev3	rev4	rev5	rev6	rev7	rev8	summ1	summ2	summ3	rating1	rating2	rating3	rating4	rating5	rating6	rating7	rating8
            if i == 0:
                header = row
                continue

            assert len(header) == len(row)
            row_dict = {header[i]: row[i] for i in range(len(header))}
            for id in range(1,9):
                review = row_dict[f'rev{id}']
                rating = row_dict[f'rating{id}']

                sample['units'].append({
                    'text': review,
                    "value": {'rating': int(float(rating))},
                })

            sample['meta'] = {'group_id': row[1], 'id': file_name, "type": "reviews", "seperator": " || ", "category": row_dict['cat']}
            sample['model_input'] = " || ".join(x['text'] for x in sample['units'])
            sample['gold_summary'] = [row_dict[f'summ{x}'] for x in range(1, 4)]
            value_input = defaultdict(list)
            for unit in sample['units']:
                value_input[unit["value"]["rating"]].append(unit['text'])
            value_input = [' || '.join(value_input[x]) for x in rating_list]
            sample['value_input'] = {'rating':value_input}
            sample['value_mapping'] = {'rating': rating_list }

            data.append(sample)


    if not os.path.exists(target_path):
        os.makedirs(target_path)
    with open(os.path.join(target_path,"amazon_gold.json"), 'w')as file:
        json.dump(data, file, indent=2, ensure_ascii=False)