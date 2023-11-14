from ThirdParty.BARTScore.bart_score import BARTScorer
from metrics.score.basic import *
class BARTScore:
    def __init__(self, device, temperature):
        self.bart_scorer = BARTScorer(device=f'cuda:{device}', checkpoint='facebook/bart-large-cnn')
        self.temperature = temperature

    def get_score(self, source, target, generate=False):
        if generate:
            bartscore, pred = self.bart_scorer.score([source], [target], batch_size=4, generate=generate)  # generation scores from the first list of texts to the second list of texts.
        else:
            bartscore = self.bart_scorer.score([source], [target], batch_size=4, generate=generate)
        # probs = [math.pow(math.e, x) for x in bartscore]
        if generate:
            return bartscore[0], pred[0]
        else:
            return bartscore[0]

    def get_processed_score(self, source:List[str], target:str)->List[float]:
        bartscores = []
        for s in source:
            score = self.bart_scorer.score([s], [target], batch_size=4)
            bartscores.append(score[0])
        softmax_bartscores = softmax(bartscores, temperature=self.temperature)
        return softmax_bartscores