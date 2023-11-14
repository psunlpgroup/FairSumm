import os.path
import os
import sys
sys.path.append("/scratch1/yfz5488/fairsumm")
os.environ['TRANSFORMERS_CACHE'] = '/scratch1/yfz5488/hf_cache'
# The local directory is too small, change to a large storage space.
# Note that this file can only caculate one dataset per pass, otherwise remember to clean the global parameters
from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
from utils.dataset import *
import argparse
from score.basic import *
from score.acu import ACUScore
from score.bert import BERTScore
from score.bart import BARTScore

CUTOFF_THRESHOLD = 1
UNFAIR_THRESHOLD = 0.8
NUM_THRESHOLD = 0
TEMPERATURE = 2 # 0.1
DEVICE=3

def collect_spoken_tokens(source):
    speaker_spoken_tokens = defaultdict(list)
    for turn in source:
        speaker = turn.split(":")[0].strip()
        words = word_tokenize(turn[len(speaker) + 3:].lower())
        words = [x for x in words if x not in ST_WORDS]
        speaker_spoken_tokens[speaker].extend(words)
    return speaker_spoken_tokens

def collect_speakers(source):
    speakers = set()
    for turn in source:
        speaker = turn.split(":")[0].strip()
        speakers.add(speaker)
    return speakers

def cutoff_distribution(source, target, cutoff=CUTOFF_THRESHOLD):
    # cut off long tails
    speakers = sorted(source.keys(), key=lambda x: -source[x])
    ratio_count = 0
    source_tot = sum(source.values())
    target_tot = sum(target.values())
    new_speakers = []
    for speaker in speakers:
        ratio_count += source[speaker] / source_tot
        new_speakers.append(speaker)
        if ratio_count >= cutoff:
            break

    # assign new speakers to data
    speakers = new_speakers
    source = {x: y for x, y in source.items() if x in speakers}
    target = {x: y for x, y in target.items() if x in speakers}

    return source, target

def compare_distribution(source, target):
    speakers = source.keys()
    source_tot = sum(source.values())
    target_tot = sum(target.values())
    diff = []

    for speaker in speakers:
        source_ratio = source[speaker] / source_tot
        target_ratio = target[speaker] / target_tot if target_tot else 1 / len(speakers)
        diff.append(max(0,source_ratio-target_ratio))

    return sum(diff)/len(diff)

def compute_second_order_fairness(value_dict):
    values = []
    for k, v in value_dict.items():
        v = [math.fabs(x) for x in v]
        values.append(sum(v)/len(v))
    # KMeans target
    avg = lambda x: sum(x)/len(x)
    values = [math.fabs(x-avg(values)) for x in values]
    return avg(values)

unfair_speaker = defaultdict(int)
unfair_distance = defaultdict(list)
def detect_unfair(source, target, threshold=UNFAIR_THRESHOLD, verbose=False):
    speakers = sorted(source.keys(), key=lambda x: -source[x])
    source = {x:y for x,y in source.items() if x in speakers}
    target = {x:y for x,y in target.items() if x in speakers}
    source_tot = sum(source.values())
    target_tot = sum(target.values())

    # count unfair
    unfair_count = 0
    unfair_speakers = []
    unfair_ratios = []
    for speaker in speakers:
        source_ratio = source[speaker] / source_tot
        target_ratio = target[speaker] / target_tot if target_tot else 1/len(speakers)
        unfair_distance[speaker].append(max(0,source_ratio-target_ratio))
        if target_ratio <= source_ratio * threshold:
            unfair_count += 1
            unfair_speaker[speaker] += 1
        if verbose:
            print(speaker, "source ratio:", round(source_ratio,2), "target ratio:", round(target_ratio,2),
                  "Unfair" if target_ratio/source_ratio < UNFAIR_THRESHOLD else "Fair")
        # if target_ratio / source_ratio < UNFAIR_THRESHOLD:
            unfair_speakers.append(speaker)
            unfair_ratios.append(source_ratio)
    if verbose:
        print(unfair_speakers, [round(x,2) for x in unfair_ratios])
        return unfair_count, unfair_speaker, unfair_ratios
    return unfair_count


def compute_new_token(source, target, synonyms=False):
    new_source = ""
    new_target = []
    if type(target) == list:
        new_target = target
    else:
        new_target = word_tokenize(target)
    if type(source) == list:
        if len(source[0]) != 1:
            for s in source:
                new_source += s
    else:
        new_source = source

    hit_count = 0
    target_hit_string = ""
    for t in new_target:
        t_list = [t]
        if synonyms:
            t_list = get_clean_sys(t, stem=False) + t_list

        if any(x in new_source for x in t_list):
            hit_count+=1
            target_hit_string += t+'# '
        else:
            target_hit_string += t+' '


    # print("source:", len(new_source), "target:", len(new_target), "new token:", len(new_target) - hit_count,
    #       "new rate:",(len(new_target) - hit_count)/len(new_target),
    #       '\ntarget:', target_hit_string)

    return (len(new_target) - hit_count), (len(new_target) - hit_count)/len(new_target)


def analysis_data(data_path, pred_path, save_path=None,
                    remove_overlap=False,
                    attribute="gender",
                    bart_generate=False,
                    use_bartscore=False,
                    use_bertscore=False,
                    use_acuscore=False,
                    skip_empty=False,
                    name=None):
    # Load data from json file
    data_source = json.load(open(data_path))
    data_target = read_list_asline(pred_path)
    if data_target[0][0] == '{': #  if it is not simply string but json
        data_target = [json.loads(x) for x in data_target]
    else:
        data_target = [{'prediction':x} for x in data_target]

    if 'court' in data_path:
        data_target = data_target[:len(data_source)]
    assert len(data_target) == len(data_source)


    #  Global Parameters
    skip = 0
    diffs = []
    unfairs = []
    new_data = []
    if use_bartscore:
        scorer = BARTScore(device=DEVICE, temperature=TEMPERATURE)
        print("BARTScorer built!")
    if use_acuscore:
        acu_scorer = ACUScore(device=DEVICE, temperature=TEMPERATURE)
        print("ACUScorer built!")
    if use_bertscore:
        bert_scorer = BERTScore(st_words=ST_WORDS, temperature=TEMPERATURE)
        print("BERTScorer built!")

    # Compute metrics for each sample and combine them together
    for id, (sample, pred) in enumerate(tqdm(zip(data_source, data_target), total=len(data_source))):

        ## Extract needed strings
        source = sample['value_input'][attribute] # source is value input
        target = pred['prediction']
        if type(target) is dict:
            target = target['text'] # some newer predictions have

        # Extreme Test: oracle
        # import random
        # # target = sample['value_input'][attribute][0]
        # male_ratio = sample['meta']['male_ratio']
        # male_samples = sample['value_input'][attribute][0]
        # female_samples = sample['value_input'][attribute][1]
        # if male_ratio == 0.2:
        #     male_samples = random.sample(male_samples.split(" || "),1)
        #     female_samples = random.sample(female_samples.split(" || "),4)
        # elif male_ratio == 0.5:
        #     male_samples = random.sample(male_samples.split(" || "), 2)
        #     female_samples = random.sample(female_samples.split(" || "), 2)
        # elif male_ratio == 0.8:
        #     male_samples = random.sample(male_samples.split(" || "), 4)
        #     female_samples = random.sample(female_samples.split(" || "), 1)
        # target = ' || '.join(male_samples + female_samples)
        #
        # # Extreme Test male_only
        # target = sample['value_input'][attribute][0]

        ## Alpaca may contain empty strings
        if not len(target.strip()) and skip_empty:
            skip += 1
            continue
        # Compute three scores first because of the original source and target
        if use_bartscore:
            print("BART Score:")
            bartscores = []
            genscores = []
            if bart_generate:
                for s in source:
                    score, pred = scorer.get_score(s, target, generate=True)
                    bartscores.append(score)
                    genscores.append(scorer.get_score(pred, target))
            else:
                for s in source:
                    score = scorer.get_score(s, target, generate=False)
                    bartscores.append(score)
                    genscores.append(1)
            # bartscore_all = scorer.get_score(source_shuffle[0], target)
            print("bartscore origin:", bartscores)
            # print("bartscore all:", bartscore_all)
            norm = lambda x: [s / sum(x) for s in x]
            softmax_bartscores = softmax(bartscores, temperature=TEMPERATURE)
            norm_bartscore = norm(bartscores)
            print("use softmax:", softmax_bartscores)
            print("use norm:", norm_bartscore)
            # bartscores = [bartscore_all - x for x in bartscores]
            # print("bartscore sub:", bartscores)
            # bartscores = [bartscores[i]/genscores[i] for i in range(len(bartscores))]
            # norm = lambda x: [s/sum(x) for s in x]
            # softmax_bartscores = softmax(bartscores)
            # norm_bartscore = norm(bartscores)
            # print("use softmax:", softmax_bartscores)
            # print("use norm:", norm_bartscore)
            target_distribution_bart = {id: x for id, x in enumerate(softmax_bartscores)}
        if use_bertscore:
            print("BERTScores:")
            bertscores = bert_scorer.get_score(source, target)#, method='sentence')
            target_distribution_bert = bertscores
            print(bertscores)
        if use_acuscore:
            print("ACUScores:")
            acuscores = acu_scorer.get_score(source, target,name=name+str(id))
            target_distribution_acu = acuscores

        ## Preprocess data and compute rule based score
        ## After this section, source and target will transform
        # print("Origin:")
        # compute_new_token(source, target)

        # Lower source and target
        source = [x.lower() for x in source]
        original_target = target
        target = target.lower()
        # print("After lower:")
        # compute_new_token(source, target)

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
        # print("After remove stop words:")
        # compute_new_token([' '.join(x) for x in list(value_tokens.values())], target)

        # now source and target are words
        # Do lemmarizing on source and target
        for key in value_tokens.keys():
            val = [get_lemma(x) for x in value_tokens[key]]
            value_tokens[key] = val
        target = [get_lemma(x) for x in target]
        # print("After lemmarizing:")
        # compute_new_token([' '.join(x) for x in list(value_tokens.values())], target)

        # Do synonyms for target (but we do not use synonyms)
        # print("After synonyms:")
        # compute_new_token([' '.join(x) for x in list(value_tokens.values())],
        #                                                    target, synonyms=True)

        # Do stemming for source and target
        for key in value_tokens.keys():
            val = [ps.stem(x) for x in value_tokens[key]]
            value_tokens[key] = val
        target = [ps.stem(x) for x in target]
        # print("After Stemming:")
        # compute_new_token([' '.join(x) for x in list(value_tokens.values())], target, synonyms=True)
        #print("intersection words:", [token for token in target if token in interset])
        if remove_overlap:
            interset = set()
            for key in value_tokens.keys():
                if len(interset) == 0:
                    interset = set(value_tokens[key])
                else:
                    interset = set(value_tokens[key]) & interset
                # print(key, len(value_tokens[key]))
            # print("intersetion:", len(interset), interset)
            # Remove overlap for target
            target = [token for token in target if token not in interset]
            # Remove overlap for source
            for key in value_tokens.keys():
                val = [x for x in value_tokens[key] if x not in interset]
                value_tokens[key] = val


        # Compute target & source distribution
        for token in target:
            for value, val_tokens in value_tokens.items():
                value_counter[value] += val_tokens.count(token) > 0
                if val_tokens.count(token):
                    value_list[value].append(token)
        #print("value list:", value_list)
        target_distribution = value_counter
        print("target (rule):", target_distribution)

        # Dicide if we need to change target_distribution
        if use_bartscore:
            target_distribution = target_distribution_bart
        elif use_bertscore:
            target_distribution = target_distribution_bert
        elif use_acuscore:
            target_distribution = target_distribution_acu

        source_distribution = {x: len(y) for x, y in value_tokens.items()}
        print("source:", source_distribution)
        print("target:", target_distribution)
        print()
        if len(source_distribution) == 0:
            skip += 1
            continue



        # Cutoff long tail distribution
        cutoff_source, cutoff_target = cutoff_distribution(source_distribution, target_distribution)
        diff = compare_distribution(cutoff_source, cutoff_target)
        diffs.append(diff)
        unfair_count = detect_unfair(cutoff_source, cutoff_target)
        unfairs.append(unfair_count > NUM_THRESHOLD)
        print("Sample's diff:", diff)
        print("Fair or not:", "Unfair" if unfair_count > 0 else "Fair")

        new_data.append({
            "source": source,  # source text, one sentence one line
            "target": target,  # preprocessed target
            "original target": original_target, # target itself
            "source distribution": source_distribution,  # a dict {value: frequency in source}
            "target distribution": target_distribution,  # a dict {value: +1 if one target token in source}
            "cutoff line": CUTOFF_THRESHOLD,  # only take top n% medical terms as value, this is n%
            "unfair line": UNFAIR_THRESHOLD,  # when <n% of source, under-represented, treat as unfair
            "cutoff source": cutoff_source,  # process source distri. with cutoff line
            "cutoff target": cutoff_target,  # process target distri. with cutoff line
            "cutoff diff": diff,  # number of absolute difference between source/target distri.
            "cutoff unfair count": unfair_count  # how many values are unfair, if > 0 we regard this sample unfair
        })


    # Global counting
    avg_diffs = sum(diffs) / len(diffs)
    avg_unfairs = sum(unfairs) / len(unfairs)
    print(data_path)
    print("average difference between summary and soruce:", avg_diffs)
    print("average ratio of samples that are unfair:", avg_unfairs)
    print("Total skipped:", skip)

    # Second Order fairness
    second_fairness = compute_second_order_fairness(unfair_distance)
    value_bur = [sum(x)/len(x) for x in unfair_distance.values()]
    print("Value level BUR is:", value_bur)
    print("Value level count is:", unfair_speaker)
    print("Second Fairness is:", second_fairness)

    all_info = {
        'BUR': avg_unfairs,
        'UER': avg_diffs,
        'SOF':{'value bur': value_bur,
               'value count': unfair_speaker,
               'all': second_fairness},
        'metric details': new_data,
        'inputs': {"source":data_source,"prediction":data_target},
        'meta':{
            "use bart": use_bartscore,
            'use acu': use_acuscore,
            'use bert': use_bertscore,
            'attribute': attribute
        }
    }
    # Save to file
    with open(save_path, 'w') as file:
        json.dump(all_info, file, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    ############################# Parameters ####################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=
        '/scratch1/yfz5488/fairsumm/preprocessing/preprocessed_datasets/claritin.json')
    parser.add_argument('--pred_path', type=str, default=
        '/scratch1/yfz5488/fairsumm/models/GPTs/results/Claritin/predictions_15.txt')
    parser.add_argument('--save_path', type=str, default=
         'results/Claritin/exp_test_bert.json')
    parser.add_argument('--attribute', type=str, default='gender')
    parser.add_argument('--name', type=str, default='claritin_test')
    parser.add_argument('--score', type=str, default='bart')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--remove_overlap', action='store_true')
    parser.add_argument('--skip_empty', action='store_true')
    args = parser.parse_args()
    ##############################################################################
    create_folder("results/Election/")
    create_folder("results/Amazon/")
    create_folder("results/Yelp/")
    create_folder("results/OxfordDebates")
    create_folder("results/SupremeCourt")

    use_bart = args.score == 'bart'
    use_bert = args.score == 'bert'
    use_acu  = args.score == 'acu'
    # skip_empty = args.skip_empty
    skip_empty = True
    DEVICE = args.device

    analysis_data(data_path=args.data_path,
                  pred_path=args.pred_path,
                  save_path=args.save_path,
                  attribute=args.attribute,
                  name=args.name,
                  remove_overlap=args.remove_overlap,
                  use_acuscore=use_acu,
                  use_bartscore=use_bart,
                  use_bertscore=use_bert,
                  skip_empty=skip_empty)

    # analysis_data(data_path="/home/yfz5488/fairsumm/preprocessing/preprocessed_datasets/claritin.json",
    #               pred_path="/home/yfz5488/fairsumm/models/GPTs/results/Claritin/predictions_16.txt",
    #               save_path="results/Claritin/exp16_rule.json",
    #               attribute='gender')

    # analysis_data(data_path="/home/yfz5488/fairsumm/preprocessing/preprocessed_datasets/claritin.json",
    #               pred_path="/home/yfz5488/fairsumm/models/GPTs/results/Claritin/predictions_16.txt",
    #               save_path="results/Claritin/exp16_bart.json",
    #               attribute='gender',
    #               use_bartscore=True)

    # analysis_data(data_path="/home/yfz5488/fairsumm/preprocessing/preprocessed_datasets/claritin.json",
    #               pred_path="/home/yfz5488/fairsumm/models/GPTs/results/Claritin/predictions_16.txt",
    #               save_path="results/Claritin/exp16_bert.json",
    #               attribute='gender',
    #               use_bertscore=True)