# nltk related operations to remove useless tokens in source
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
from nltk import FreqDist
from nltk.corpus import brown
from string import punctuation
import os, json, math
from typing import List

nltk.download('popular')
nltk.download('brown')
lemma = WordNetLemmatizer()
ps = PorterStemmer()

ST_WORDS = [p for p in punctuation] + stopwords.words() + ['||', "people", "''", '``', '...', '..']
MASK_WORDS = ["subject", "subjects", "father", "||"]
FREQUENT_TOP_N = 2000


def get_synonyms(word, pos):
    # Gets word synonyms for part of speech '
    for synset in wn.synsets(word, pos=pos_to_wordnet_pos(pos)):
        for lemma in synset.lemmas():
            yield lemma.name()


def pos_to_wordnet_pos(penntag, returnNone=False):
    # Mapping from POS tag word wordnet pos tag '
    morphy_tag = {'NN': wn.NOUN, 'JJ': wn.ADJ,
                  'VB': wn.VERB, 'RB': wn.ADV}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return None if returnNone else ''


def get_tag(token):
    result = nltk.pos_tag([token])
    word, tag = result[0]
    return tag


def get_lemma(token):
    pos = pos_to_wordnet_pos(get_tag(token))
    if pos == "":
        lem_token = lemma.lemmatize(token)
    else:
        lem_token = lemma.lemmatize(token, pos=pos)
    # print(token, pos, lem_token)
    return lem_token


def get_clean_sys(token, stem=False):
    result = nltk.pos_tag([token])
    word, tag = result[0]
    # Filter for unique synonyms not equal to word and sort.
    syns = get_synonyms(word, tag)
    unique = [synonym.lower() for synonym in syns]
    if stem:
        unique = [ps.stem(x) for x in unique]
    return unique


# Get frequent dict
def get_frequent_words(top_n=FREQUENT_TOP_N):
    target_file = "frequent_words.json"
    if os.path.exists(target_file):
        return json.load(open(target_file))
    else:
        frequency_list = FreqDist(i.lower() for i in brown.words())
        frequent_words = [x[0] for x in frequency_list.most_common()[:top_n]]
        frequent_words += MASK_WORDS
        with open(target_file, 'w') as file:
            json.dump(frequent_words, file, indent=2)
        return frequent_words


COMMON_WORDS = get_frequent_words()

### New operators for results

def softmax(logits, temperature=0.1):
    if temperature == 0:
        max_val = max(logits)
        logits = [x==max_val for x in logits]
        prob = [x / sum(logits) for x in logits]
        return prob
    logits = [math.pow(math.e, x / temperature) for x in logits]
    prob = [x / sum(logits) for x in logits]
    return prob

def norm(logits):
    if sum(logits) == 0:
        return [1/len(logits) for x in logits]
    return [x/sum(logits) for x in logits]

### Tools for convient
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


### Compute Soruce Distribution
from collections import defaultdict
from nltk import word_tokenize
def get_source_distribution(value_source:List[str]):
    source = [x.lower() for x in value_source]
    value_tokens = {}
    for idx, sent in enumerate(source):
        words = word_tokenize(sent.lower())
        words = [x for x in words if x not in ST_WORDS]
        value_tokens[idx] = words
        for key in value_tokens.keys():
            val = [get_lemma(x) for x in value_tokens[key]]
            value_tokens[key] = val
            val = [ps.stem(x) for x in value_tokens[key]]
            value_tokens[key] = val
    source_distribution = {x: len(y) for x, y in value_tokens.items()}
    return source_distribution