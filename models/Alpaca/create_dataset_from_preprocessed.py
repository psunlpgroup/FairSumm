import json
datasets = ['amazon','claritin','election','oxforddebates','supremecourt','yelp']

for dataset in datasets:
    # load raw
    path = "/home/yfz5488/fairsumm/processed_datasets/{}.json".format(dataset)
    with open(path) as file:
        raw_data = json.load(file)
    # pick useful data
    data = []
    # amazon: model_input
    # claritin: input
    # election: input
    # oxforddebates: input.join('\n')
    # supermecourt: input.join('\n')
    # yelp:model_input
    for sample in raw_data:
        if dataset in ['amazon', 'yelp']:
            if 'model_input' not in sample.keys():
                print(sample)
            data.append(sample['model_input'])
        elif dataset in ['claritin', 'election']:
            data.append(sample['input'])
        else:
            data.append('\n'.join(sample['input']))

    # save to target
    with open('for Nan/{}.json'.format(dataset), 'w') as file:
        json.dump(data,file,indent=2,ensure_ascii=False)