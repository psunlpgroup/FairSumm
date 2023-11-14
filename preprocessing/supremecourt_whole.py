# Use this script, we can get dataset of various formats

from dataset_loader import SupremeCourt
import json
from collections import defaultdict
new_data = []
loader = SupremeCourt.Loader()
# if wanna get whole dataset max_len = 20000000, collection > 204
# if wanna get 665 dataset max_len = 2000, collection = 100
# if wanna get 1365 extend max_len = 2000, collection > 204
data,_,ids = loader.load(collection=2000,max_len=2000, verbose=True, dataset_path="/home/yfz5488/fairsumm/datasets/supcourt/supreme_court_dialogs_corpus_v1.01")
for sample, id, in zip(data['test'], ids):
    new_data.append({
        'input': sample,
        'id': id
    })

# create dataset
dataset = []
for sample in new_data:
    # Get speaker and value list
    values = defaultdict(list)
    for turn in sample['input']:
        content = turn.split(':')[1].strip()
        speaker = turn.split(':')[0].strip()
        values[speaker].append(turn)

    value_list = list(values.keys())
    value_as_input = ["\n".join(values[x]) for x in value_list]

    units = []
    for turn in sample['input']:
        content = turn.split(':')[1].strip()
        speaker = turn.split(':')[0].strip()
        units.append({"text": turn, "value": {"speaker": speaker}, "content":content})


    # Append to dataset
    dataset.append({
        "model_input": "\n".join(sample['input']),
        "value_input": {'speaker': value_as_input},
        "value_mapping": {'speaker': value_list},
        "meta": {"type": "dialogue",
                 "seperator": "\n"},
        "units": units
    })  
print("Total samples:", len(dataset))
json.dump(dataset, open("preprocessed_datasets/supremecourt_extend.json",'w'), indent=2, ensure_ascii=False)
