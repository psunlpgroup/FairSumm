from dataset_loader import OxfordDebates
import json
from collections import defaultdict
new_data = []
loader = OxfordDebates.Loader()
data,_,ids = loader.load(verbose=True, dataset_path="/home/yfz5488/fairsumm/datasets/Oxford-style debates")
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
        "model_input": "\n".join(sample),
        "value_input": {'speaker': value_as_input},
        "value_mapping": {'speaker': value_list},
        "meta": {"type": "dialogue",
                 "seperator": "\n"},
        "units": units
    })
print("Total samples:", len(dataset))
json.dump(dataset, open("preprocessed_datasets/oxforddebates.json",'w'), indent=2, ensure_ascii=False)

# Previous one is to generate a segmented dataset
# This one is to generate a dataset with whole input
#
# dataset = []
# loader = OxfordDebates.Loader()
# data,_,ids = loader.load(verbose=True,max_len=1500000,collection=1000, dataset_path="/home/yfz5488/fairsumm/datasets/Oxford-style debates")
# for sample, id, in zip(data['test'], ids):
#     dataset.append({
#         'input': sample,
#         'id': id
#     })
#
# json.dump(dataset, open("/home/yfz5488/fairsumm/processed_datasets/oxforddebates_whole.json",'w'), indent=2, ensure_ascii=False)
#
