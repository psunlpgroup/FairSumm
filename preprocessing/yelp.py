# Preprocess the raw amazon dataset to the format that can be run
import json
import os
import csv
import random
random.seed(42)
rating_list = [1,2,3,4,5]
sentiment_list = ['pos', 'neu', 'neg']
TOTAL_SAMPLED = 1500
from collections import defaultdict

from nltk.sentiment.vader import SentimentIntensityAnalyzer

def get_sent(sen):
    # Input a review, return the sentiment
    sid = SentimentIntensityAnalyzer()

    sentiment = sid.polarity_scores(sen)
    max_sentiment = -1
    senti = None
    for k in sentiment:
        if k == 'compound' or k == 'neu':
            continue
        if sentiment[k] > max_sentiment:
            max_sentiment = sentiment[k]
            senti = k
    return senti if max_sentiment != 0 else "neu"

if __name__ == '__main__':
    folder_path = "../datasets/yelp/reviews/val"
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
                "value": {'rating': int(float(row[2])), 'sentiment': get_sent(row[1])},
                'category': row[3],
            })

        sample['meta'] = {'id': file_name, "type": "tweets", "seperator": " || "}
        sample['model_input'] = " || ".join(x['text'] for x in sample['units'])

        # deal with ratings
        value_input_rating = defaultdict(list)
        for unit in sample['units']:
            value_input_rating[unit["value"]["rating"]].append(unit['text'])
        value_input_rating = [' || '.join(value_input_rating[x]) for x in rating_list]

        # deal with sentiment
        value_input_sentiment = defaultdict(list)
        for unit in sample['units']:
            value_input_sentiment[unit["value"]["sentiment"]].append(unit['text'])
        value_input_sentiment = [' || '.join(value_input_sentiment[x]) for x in sentiment_list]

        sample['value_input'] = {'rating':value_input_rating, 'sentiment':value_input_sentiment}
        sample['value_mapping'] = {'rating': rating_list, 'sentiment': sentiment_list}

        data.append(sample)
        if len(data) >= TOTAL_SAMPLED:
            break

    if not os.path.exists(target_path):
        os.makedirs(target_path)
    with open(os.path.join(target_path,"yelp.json"), 'w')as file:
        json.dump(data, file, indent=2, ensure_ascii=False)