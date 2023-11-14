from ThirdParty.AutoACU.autoacu import A2CU
from collections import defaultdict

def softmax(logits):
    import math
    logits = [math.pow(math.e, x*10) for x  in logits]
    prob = [x/sum(logits) for x in logits]
    return prob

class ACUScore():
    def __int__(self, device):
        self.a2cu = A2CU(device=device)  # the GPU device to use

    def get_score(self, source, target):
        cands = [target for _ in range(len(source))]
        refs = source
        prec_scores = self.a2cu.score(
            references=refs,
            candidates=cands,
            generation_batch_size=2,  # the batch size for ACU generation
            matching_batch_size=16,  # the batch size for ACU matching
            output_path="tmp_eval.txt",  # the path to save the evaluation results
            precision_only=True,  # whether to only compute the recall score
            acu_path="tmp_acu.txt"  # the path to save the generated ACUs
        )
        P = softmax(prec_scores)
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
