from ThirdParty.AutoACU.autoacu import A2CU
from collections import defaultdict
from .basic import softmax, norm
from time import time

class ACUScore:
    def __init__(self, device, temperature):
        self.a2cu = A2CU(device=device) # the GPU device to use
        self.temperature = temperature

    def get_score(self, source, target, name=None):
        cands = [target for _ in range(len(source))]
        refs = source
        if name is None:
            name = time()
        prec_scores = self.a2cu.score(
            references=refs,
            candidates=cands,
            generation_batch_size=2,  # the batch size for ACU generation
            matching_batch_size=16,  # the batch size for ACU matching
            output_path=f"tmp/tmp_eval_{name}.txt",  # the path to save the evaluation results
            precision_only=True,  # whether to only compute the recall score
            acu_path=f"tmp/tmp_acu{name}.txt"  # the path to save the generated ACUs
        )
        P = softmax(prec_scores, self.temperature)
        # P = norm(prec_scores)  # Remove 0
        token_precisions = defaultdict(list)
        for i in range(len(cands)):
            token_precisions[i].append(P[i])

        # Compute final scores
        scores = defaultdict(float)
        scores_sum = 0
        avg = lambda x: sum(x) / len(x)
        for i, s in token_precisions.items():
            scores[i] = avg(s)
            scores_sum += scores[i]
        # Normalization
        for i, s in scores.items():
            scores[i] = scores[i] / scores_sum
        return scores

if __name__ == '__main__':
    sc = ACUScore(device=1, temperature=0.1)