import gzip
import os
import json
import random
from collections import defaultdict
random.seed(42)
class Loader():
    def __init__(self):
        super().__init__()
        self.data = {"train": [], "test": [], "val": []}
        self.label = {"train": [], "test": [], "val": []}
        self.value = {"train": [], "test": [], "val": []}
        self.is_dialogue = False
        self.name = "Election"

    def load(self, dataset_path, num_sample=100, num_tweets=50, repu_ratio=0.4, demo_ratio=0.4, min_length=10, shuffle=True, verbose=False):
        samples = defaultdict(list)
        for data_type in ['test']: # we treat every sentence as test set
            if 'processed_datasets' in dataset_path: # Load from processed json
                with open(dataset_path) as file:
                    data = json.load(file)
                for sample in data:
                    if not shuffle:
                        source = sample['value_input']['politics']
                    else:
                        source = [x['text'] for x in sample['units']]
                        value = [x['value']['politics'] for x in sample['units']]
                    summary = ""  # we do not use the summary
                    self.data[data_type].append(source)
                    self.label[data_type].append(summary)
                    self.value[data_type].append(value)
            else:
                # Read raw tweets
                tweets = []
                with open(dataset_path+'input.txt', encoding='utf-8') as file:
                    for line in file:
                        line = line.strip().split("<||>")
                        tweets.append({
                            'id':line[0],
                            'prefer': line[1],
                            'text': line[2]
                        })

                # Create a dataset by sampling
                men_tweets = [t for t in tweets
                              if t['prefer'] == 'Pro-Republican' and len(t['text']) > min_length]
                women_tweets = [t for t in tweets
                              if t['prefer'] == 'Pro-Democrat' and len(t['text']) > min_length]
                neutral_tweets = [t for t in tweets
                              if t['prefer'] == 'Neutral' and len(t['text']) > min_length]

                for i in range(num_sample):
                    men_collection = [x['text'] for x in random.sample(men_tweets,int(num_tweets*repu_ratio))]
                    women_collection = [x['text'] for x in random.sample(women_tweets, int(num_tweets*demo_ratio))]
                    neutral_collection = [x['text'] for x in random.sample(neutral_tweets, num_tweets - int(num_tweets*repu_ratio) - int(num_tweets*demo_ratio))]

                    source = [' || '.join(men_collection), ' || '.join(women_collection), ' || '.join(neutral_collection)]
                    if shuffle:
                        collections = men_collection + women_collection + neutral_collection
                        random.shuffle(collections)
                        source = [' || '.join(collections)]
                    if verbose:
                        samples[data_type].append([men_collection, women_collection, neutral_collection])

                    summary = "" # we do not use the summary
                    self.data[data_type].append(source)
                    self.label[data_type].append(summary)
        if verbose:
            return self.data, self.label, samples
        return self.data, self.label, self.value



if __name__ == '__main__':
    loader = Loader()
    loader.load(dataset_path="/home/yfz5488/fairsumm/datasets/FairSumm-master/Dataset/US-Election/")