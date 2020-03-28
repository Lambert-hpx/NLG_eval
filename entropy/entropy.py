import numpy as np
from collections import defaultdict
class Entropy:
    def __init__(self):
        self.etp_score = [0.0,0.0,0.0,0.0]
        self.counter = [defaultdict(int),defaultdict(int),defaultdict(int),defaultdict(int)]

    def compute_score(self, gts, res):
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()
        for id in imgIds:
            hypo = res[id][0]
            words = hypo.strip('\n').split()
            for n in range(4):
                for idx in range(len(words)-n):
                    ngram = ' '.join(words[idx:idx+n+1])
                    self.counter[n][ngram] += 1
        for n in range(4):
            total = sum(self.counter[n].values())
            for v in self.counter[n].values():
                self.etp_score[n] += - v /total * (np.log(v) - np.log(total))
            self.etp_score[n] = float(self.etp_score[n])
        return float(self.etp_score[3]),self.etp_score

    def method(self):
        return "Entropy"