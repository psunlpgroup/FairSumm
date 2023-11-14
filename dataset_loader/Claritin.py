import gzip
import os
import json
import random

import nltk

from utils.dataset import *
random.seed(42)
class Loader():
    def __init__(self):
        super().__init__()
        self.data = {"train": [], "test": [], "val": []}
        self.label = {"train": [], "test": [], "val": []}
        self.is_dialogue = False
        self.name = "Claritin"

    def clear(self):
        self.data = {"train": [], "test": [], "val": []}
        self.label = {"train": [], "test": [], "val": []}
        self.is_dialogue = False
        self.name = "Claritin"

    def n_sample(self, data, n, k, shuffle=True):
        """
        Sample data points from a large dataset.
        :param data: large dataset, containing reviews (~4k reviews)
        :param n: number of samples we want to get
        :param k: number of reviews in each sample
        :param shuffle: shuffle dataset before sample or not
        :return: a list of samples, each sample contains several reviews
        """
        if shuffle:
            random.shuffle(data)

        # Expand data in case there are not enough samples
        repeat = n*k // len(data) + 1
        data = data * repeat

        # Get samples
        splits = data[:n*k]
        samples = []
        for i in range(n):
            samples.append(splits[k*i:k*(i+1)])
        return samples

    def load(self, dataset_path, num_sample=100, num_tweets=50, male_ratio=0.5, min_length=10, max_length=70, shuffle=False, verbose=False):
        for data_type in ['test']: # we treat every sentence as test set
            if 'processed_datasets' in dataset_path: # Load from processed json
                with open(dataset_path) as file:
                    data = json.load(file)
                for sample in data:
                    if shuffle:
                        source = sample['reviews']
                    else:
                        source = sample['values']
                    summary = ""  # we do not use the summary
                    self.data[data_type].append(source)
                    self.label[data_type].append(summary)
            else:
                # Read raw tweets
                tweets = []
                with open(dataset_path+'input.txt', encoding='utf-8') as file:
                    for line in file:
                        line = line.strip().split("<||>")
                        tweets.append({
                            'id':line[0],
                            'gender': line[1],
                            'text': line[2]
                        })

                # Create a dataset by sampling
                men_tweets = [t for t in tweets
                              if t['gender'] == 'male' and min_length <= len(nltk.word_tokenize(t['text'])) <= max_length]
                women_tweets = [t for t in tweets
                                if t['gender'] == 'female' and min_length <= len(nltk.word_tokenize(t['text'])) <= max_length]

                men_tweets = self.n_sample(men_tweets, num_sample, int(num_tweets*male_ratio))
                women_tweets = self.n_sample(women_tweets, num_sample, num_tweets - int(num_tweets*male_ratio))

                values_input = []
                for i in range(num_sample):
                    men_collection = [x['text'] for x in men_tweets[i]]
                    women_collection = [x['text'] for x in women_tweets[i]]
                    source = [' || '.join(men_collection), ' || '.join(women_collection)]
                    if shuffle:
                        collections = men_collection + women_collection
                        random.shuffle(collections)
                        source = [' || '.join(collections)]
                    values_input.append([men_collection, women_collection])
                    summary = "" # we do not use the summary
                    self.data[data_type].append(source)
                    self.label[data_type].append(summary)
        if verbose:
            return self.data, self.label, values_input
        return self.data, self.label



if __name__ == '__main__':
    # loader = Loader()
    # data = loader.load(dataset_path="/home/yfz5488/fairsumm/datasets/FairSumm-master/Dataset/Claritin/")
    # data = [x[0] for x in data[0]['test']]
    # json.dump(data, open("claritin_freeze.json",'w'))
    loader = Loader()
    data = loader.load(dataset_path="/home/yfz5488/fairsumm/processed_datasets/claritin.json", shuffle=True)
    print(data)