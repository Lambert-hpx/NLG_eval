import numpy as np
class Avglen:
    def __init__(self):
        self.l = []

    def compute_score(self, gts, res):
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()
		
        for id in imgIds:
            hypo = res[id][0]
            self.l.append(len(hypo.strip('\n').split()))
		
        return float(np.mean(self.l)),None

    def method(self):
        return "Avglen"