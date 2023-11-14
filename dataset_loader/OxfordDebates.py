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
        self.name = "OxfordDebates"

    def clear(self):
        self.data = {"train": [], "test": [], "val": []}
        self.label = {"train": [], "test": [], "val": []}
        self.is_dialogue = True
        self.name = "OxfordDebates"

    def load(self, dataset_path, collection=10000, max_len=1500, verbose=False, first_seg=False):
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
                data = json.load(open(os.path.join(dataset_path,"iq2_data_release.json")))
                for id, sample in enumerate(data.values()):
                    transcript = sample['transcript']
                    meeting = []
                    count = 0
                    for turn in transcript:
                        speaker = turn['speaker']
                        sent = speaker + ' : ' + ' '.join(turn['paragraphs'])
                        count += len(word_tokenize(sent))
                        if count > max_len:
                            if first_seg:
                                break
                            self.data[data_type].append(meeting)
                            ids.append(id)
                            meeting = []
                            count = 0
                        meeting.append(sent)
                    if len(meeting):
                        self.data[data_type].append(meeting)
                        ids.append(id)
                self.data[data_type] = self.data[data_type][:collection]
                self.label[data_type] = self.label[data_type][:collection]
                ids = ids[:collection]
        if verbose:
            return self.data, self.label, ids
        return self.data, self.label


if __name__ == '__main__':
    loader = Loader()
    data = loader.load(verbose=True, dataset_path="/home/yfz5488/fairsumm/datasets/Oxford-style debates")
    print(len(data[0]['test']))