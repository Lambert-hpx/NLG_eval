import numpy as np
from collections import defaultdict
class Dist:
    def __init__(self):
        self.tokens = [0.0,0.0]
        self.types = [defaultdict(int),defaultdict(int)]
    
    def compute_score(self, gts, res):
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()
        for id in imgIds:
            hypo = res[id][0]
            words = hypo.strip('\n').split()
            for n in range(2):
                for idx in range(len(words)-n):
                    ngram = ' '.join(words[idx:idx+n+1])
                    self.types[n][ngram] = 1
                    self.tokens[n] += 1
        dist_list=[]
        div1 = len(self.types[0].keys())/self.tokens[0]
        dist_list.append(div1)
        div2 = len(self.types[1].keys())/self.tokens[1]
        dist_list.append(div2)
        #print(dist_list)
        return dist_list,dist_list

    def method(self):
        return "Dist"