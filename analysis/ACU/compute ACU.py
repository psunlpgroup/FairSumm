import json
import os
from metrics.score.bur_uer import BUR_UER

def read_data(path):
    turbo = json.load(open(path))

    all_target_dist = [t['target distribution'] for t in turbo['metric details']]
    all_source_dist = [s['source distribution'] for s in turbo['metric details']]
    return all_source_dist, all_target_dist

def compute_AUC(all_source_dist, all_target_dist):

    all_bur = []
    for cut_off in range(0, 11):
        cut_off = cut_off/10
        bur = []
        for t_d, s_d in zip(all_target_dist, all_source_dist):
            bur_scorer = BUR_UER(CUTOFF_THRESHOLD=cut_off)
            final_uer, final_bur = bur_scorer.compute_score(s_d, t_d)
            bur.append(final_bur)
        avg = lambda x: sum(x)/len(x)
        # print("cutoff threshold is:", cut_off, "Final BUR is:", avg(bur))
        all_bur.append(avg(bur))
    print("Final BUR are:", all_bur)
    auc = 0
    for i in range(1, 11):
        auc += (all_bur[i] + all_bur[i-1]) * 0.1 /2
    print("auc is:", auc)
    return auc

if __name__ == '__main__':
    dir = "/scratch1/yfz5488/fairsumm/metrics/results/OxfordDebates"
    result_dict = {}
    for name in os.listdir(dir):
        print(name)
        all_source_dist, all_target_dist = read_data(os.path.join(dir, name))
        acu = compute_AUC(all_source_dist, all_target_dist)
        result_dict[name] = acu

    print(json.dumps(result_dict, indent=2))




