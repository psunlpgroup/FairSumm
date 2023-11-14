from typing import List, Dict
from collections import defaultdict
from metrics.score.basic import *
from nltk import word_tokenize
from nltk.stem import PorterStemmer

class RuleScore:
    def __init__(self):
        self.ps = PorterStemmer()
    def get_score(self,source: List[str], target: str)-> Dict[int,float]:
        # Lower source and target
        source = [x.lower() for x in source]
        target = target.lower()

        # each line of source is a value
        # Word tokenization + remove stop words
        value_counter = defaultdict(int)
        value_list = defaultdict(list)
        value_tokens = {}
        for idx, sent in enumerate(source):
            words = word_tokenize(sent.lower())
            words = [x for x in words if x not in ST_WORDS]
            value_tokens[idx] = words
        target = word_tokenize(target)
        target = [token for token in target if token not in ST_WORDS]

        # now source and target are words
        # Do lemmarizing on source and target
        for key in value_tokens.keys():
            val = [get_lemma(x) for x in value_tokens[key]]
            value_tokens[key] = val
        target = [get_lemma(x) for x in target]

        # Do stemming for source and target
        for key in value_tokens.keys():
            val = [self.ps.stem(x) for x in value_tokens[key]]
            value_tokens[key] = val
        target = [self.ps.stem(x) for x in target]

        # Compute target & source distribution
        for token in target:
            for value, val_tokens in value_tokens.items():
                value_counter[value] += val_tokens.count(token) > 0
                if val_tokens.count(token):
                    value_list[value].append(token)
        target_distribution = value_counter
        # source_distribution = {x: len(y) for x, y in value_tokens.items()}
        return target_distribution