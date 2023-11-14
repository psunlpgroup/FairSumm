import json
from collections import defaultdict
#  TODO: check why should the number of samples not aligned

# parse raw data
path = "/home/yfz5488/fairsumm/models/GPTs/results/SupremeCourt/predictions_3.txt"
data = []

with open(path) as file:
    for line in file:
        data.append(json.loads(line.strip()))

new_data = []
for sample in data:
    input = sample['input'].split('\n')
    input = input[2:-2]
    new_data.append(input)

# create dataset
dataset = []
for sample in new_data:
    # Get speaker and value list
    values = defaultdict(list)
    for turn in sample:
        content = turn.split(':')[1].strip()
        speaker = turn.split(':')[0].strip()
        values[speaker].append(turn)

    value_list = list(values.keys())
    value_as_input = ["\n".join(values[x]) for x in value_list]

    units = []
    for turn in sample:
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
json.dump(dataset, open("preprocessed_datasets/supremecourt.json",'w'), indent=2, ensure_ascii=False)