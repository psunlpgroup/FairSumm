from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os, nltk
import json
import collections
import csv
nltk.download('vader_lexicon')
from utils.dataset import *

def nltk_sent(view, analysis=None):
    sid = SentimentIntensityAnalyzer()
    view_sen = []
    for sen in view:
        sentiment = sid.polarity_scores(sen)
        max_sentiment = -1
        senti = None
        for k in sentiment:
            if k == 'compound' or k == 'neu':
                continue
            if sentiment[k] > max_sentiment:
                max_sentiment = sentiment[k]
                senti = k
        view_sen.append(senti if max_sentiment != 0 else "neu")

    # cluster them
    from collections import defaultdict
    cluster = defaultdict(list)
    counter = defaultdict(int)
    for vi, sen in zip(view, view_sen):
        cluster[sen].append(vi)
        counter[sen] += 1


    if analysis == "avg":
        min_cnt = 1000000
        for key in cluster.keys():
            min_cnt = min(len(cluster[key]), min_cnt)
        for key in cluster.keys():
            cluster[key] = cluster[key][:min_cnt]
            counter[key] = len(cluster[key])
    print(counter)
    return ['\n'.join(cluster['pos']), '\n'.join(cluster['neg'])]

def cluster_source(source, ratings):
    cluster = collections.defaultdict(list)
    for s, r in zip(source, ratings):
        cluster[r].append(s)

    cluster = sorted(cluster.items(), key=lambda x: x[0])
    new_cluster = [' '.join(x[1]) for x in cluster]
    new_ratings = [x[0] for x in cluster]

    return new_cluster, new_ratings

class Loader():
    def __init__(self):
        super().__init__()
        self.data = {"train": [], "test": [], "val": []}
        self.label = {"train": [], "test": [], "val": []}
        self.ratings = {"train": [], "test": [], "val": []}
        self.is_dialogue = False
        self.name = "Fewsum"

    def load(self, dataset_path, data_name, cluster=None, add_id=True):
        self.name = self.name + "_" + data_name
        if "processed_datasets" in dataset_path: # load from processed dataset
            with open(dataset_path) as file:
                data = json.load(file)
            data_type = 'test'
            for sample in data:
                source, summary, ratings = [], [], []
                for review in sample['units']:
                    source.append(review['text'])
                    if 'amazon' in data_name.lower():
                        ratings.append(review['value']['rating'])
                    else:
                        ratings.append(review['value']['sentiment'])
                if add_id:
                    new_source = []
                    for id, rev in enumerate(source):
                        new_source.append(f"review: {rev}")
                    source = new_source

                if cluster is not None:
                    if cluster == 'sentiment':
                        source = nltk_sent(source)
                    elif cluster == 'rating':
                        source, ratings = cluster_source(source, ratings)
                    else:
                        raise NotImplementedError()

                self.data[data_type].append(source)
                self.label[data_type].append(summary)
                self.ratings[data_type].append(ratings)
        else:
            for data_type in ['train','test','val']:
                with open(os.path.join(dataset_path, data_type+'.csv')) as file:
                    reader = csv.reader(file, delimiter='\t')
                    for i, line in enumerate(reader):
                        if i == 0:
                            header = line
                            continue
                        source, summary, ratings = [], [], []
                        for key, value in zip(header, line):
                            if 'rev' in key:
                                source.append(value)
                            if 'summ' in key:
                                summary.append(value)
                            if "rating" in key:
                                ratings.append(value)

                        if add_id:
                            new_source = []
                            for id, rev in enumerate(source):
                                new_source.append(f"review: {rev}\n")
                            source = new_source

                        if cluster is not None:
                            if cluster == 'sentiment':
                                source = nltk_sent(source)
                            elif cluster == 'rating':
                                source, ratings = cluster_source(source, ratings)
                            else:
                                raise NotImplementedError()

                        self.data[data_type].append(source)
                        self.label[data_type].append(summary[0]) # we pick three of one summary
                        self.ratings[data_type].append(ratings)

        # we include all samples in test set
        self.data['test'] = self.data['test'] + self.data['val'] + self.data['train']
        self.label['test'] = self.label['test'] + self.label['val'] + self.label['train']
        self.ratings['test'] = self.ratings['test'] + self.ratings['val'] + self.ratings['train']

        return self.data, self.label, self.ratings

if __name__ == '__main__':
    loader = Loader()
    # data = loader.load(dataset_path="/home/yfz5488/fairsumm/opinion_summ/artifacts/amazon/gold_summs",
    #             data_name="amazon")

    data = loader.load(dataset_path="/home/yfz5488/fairsumm/processed_datasets/amazon.json",
                data_name="amazon")
    print(data)
