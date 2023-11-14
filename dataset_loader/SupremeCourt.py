import gzip
import os
import json
from collections import defaultdict
from utils.dataset import *
from nltk import word_tokenize

class Loader():
    def __init__(self):
        super().__init__()
        self.data = {"train": [], "test": [], "val": []}
        self.label = {"train": [], "test": [], "val": []}
        self.is_dialogue = True
        self.name = "SupremeCourt"

    def load(self, dataset_path, collection=10000, max_len=2000, verbose=False):
        if "processed_datasets" in dataset_path: # load from processed dataset
            with open(dataset_path) as file:
                data = json.load(file)
            data_type = 'test'
            for sample in data:
                source, summary = [], []
                for review in sample['units']:
                    source.append(review['text'])

                self.data[data_type].append(source)
                self.label[data_type].append(summary)
        else:
            for data_type in ['test']:
                ids = []
                samples = defaultdict(list)
                for line in open(os.path.join(dataset_path,"supreme.conversations.txt")):
                    line = [x.strip() for x in line.split(" +++$+++ ")]
                    samples[line[0]].append(line)

                for id, sample in enumerate(list(samples.values())[:collection]):
                    start = 0
                    meeting = []
                    count = 0
                    for turn in sample:
                        sent = turn[3] + " : " + turn[-1]
                        count += len(word_tokenize(sent))
                        if count > max_len:
                            self.data[data_type].append(meeting)
                            ids.append(id)
                            meeting = []
                            count = 0
                        meeting.append(sent)
                    if len(meeting):
                        self.data[data_type].append(meeting)
                        ids.append(id)
                # self.data[data_type] = self.data[data_type][:collection]
                # self.label[data_type] = self.label[data_type][:collection]
                # ids = ids[:collection]
                self.data[data_type] = self.data[data_type]
                self.label[data_type] = self.label[data_type]
                ids = ids
        if verbose:
            return self.data, self.label, ids
        return self.data, self.label


if __name__ == '__main__':
    loader = Loader()
    data = loader.load(verbose=True, dataset_path="/home/yfz5488/fairsumm/datasets/supcourt/supreme_court_dialogs_corpus_v1.01/")
    print(len(data[0]['test']))