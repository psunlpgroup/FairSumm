from ThirdParty.bert_score.bert_score import BERTScorer
from collections import defaultdict
from nltk import word_tokenize, sent_tokenize
from .basic import softmax


class BERTScore:
    def __init__(self, st_words, temperature,device=1):
        self.st_words = st_words
        self.temperature = temperature
        self.bert_score = BERTScorer(device=device,rescale_with_baseline=True, lang='en')

    def get_score(self, source, target, method="summary_combine", add_softmax=True):
        split_reviews = []
        for value in source:
            split_reviews.append([x.strip() for x in value.split("||")])
        token_precisions = defaultdict(list)
        if method == 'token':
            for token in word_tokenize(target):
                if token.lower() in self.st_words:
                    continue
                single_cands = [token for _ in range(len(split_reviews))]
                multi_refs = split_reviews
                P_mul, R_mul, F_mul = self.bert_score.score(single_cands, multi_refs)
                for i in range(len(single_cands)):
                    token_precisions[i].append(P_mul[i])
        elif method == "sentence":
            for token in sent_tokenize(target):
                single_cands = [token for _ in range(len(split_reviews))]
                multi_refs = split_reviews
                P_mul, R_mul, F_mul = self.bert_score.score(single_cands, multi_refs)
                if add_softmax:
                    P_mul = softmax(P_mul, temperature=self.temperature)
                for i in range(len(single_cands)):
                    token_precisions[i].append(P_mul[i])
        elif method == "summary_combine":
            cands = [target for _ in range(len(source))]
            refs = source
            P, R, F1 = self.bert_score.score(cands, refs)
            if add_softmax:
                P = softmax(P, temperature=self.temperature)
            for i in range(len(cands)):
                token_precisions[i].append(P[i])
        elif method == "token_combine":
            for token in word_tokenize(target):
                if token.lower() in self.st_words:
                    continue
                cands = [token for _ in range(len(split_reviews))]
                refs = [' '.join(x) for x in split_reviews]
                P_mul, R_mul, F_mul = self.bert_score.score(cands, refs)
                for i in range(len(cands)):
                    token_precisions[i].append(P_mul[i])

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
