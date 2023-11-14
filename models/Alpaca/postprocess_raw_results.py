import os,json

if __name__ == '__main__':
    folder = "results/raw"
    source_folder = "for Nan"
    target_folder = "results/postprocessed"
    for name in os.listdir(folder):
        print(name)
        path = os.path.join(folder,name)
        with open(path) as file:
            results = json.load(file)
        source = json.load(open(source_folder + '/' + name))
        all_results = []
        for s, t in zip(source, results):
            all_results.append({'input': s, 'prediction': t})

        with open(target_folder+'/'+ name, 'w') as file:
            for result in all_results:
                file.write(json.dumps(result, ensure_ascii=False)+'\n')


