from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from avglen.avglen import Avglen
from entropy.entropy import Entropy
from dist.dist import Dist
from nist.nist import Nist
class TOCEvalCap:
    def __init__(self, gts, res):
        self.gts = gts
        self.res = res

    def evaluate(self,  verbose=True):
        output = {}
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Dist(),["Dist_1","Dist_2"]),
            (Nist(),["NIST_2","NIST_4"]),
            # (Rouge(), "ROUGE_L"),
            (Meteor(),"METEOR"),
            # (Cider(), "CIDEr"),
            (Avglen(),"Avglen"),
            (Entropy(),"Entropy"),

        ]
        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            # print 'computing %s score...'%(scorer.method())
            score, scores = scorer.compute_score(self.gts, self.res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    #print(sc,scs,m)
                    if verbose:
                        print("%s: %0.5f"%(m, sc))
                    # output.append(sc)
                    output[m] = sc
            else:
                if verbose:
                    print("%s: %0.5f"%(method, score))
                # output.append(score)
                output[method] = score
        return output
def evaluate_predictions(target_src, decoded_text):
    assert len(target_src) == len(decoded_text)
    eval_targets = {}
    eval_predictions = {}
    for idx in range(len(target_src)):
        eval_targets[idx] = [target_src[idx]]
        eval_predictions[idx] = [decoded_text[idx]]

    TOCEval = TOCEvalCap(eval_targets, eval_predictions)
    scores = TOCEval.evaluate()
    return scores

target=["good what are your hobbies","yeah i 've 3 jobs then","good what are your hobbies","yeah i 've 3 jobs then","good what are your hobbies","yeah i 've 3 jobs then","good what are your hobbies","yeah i 've 3 jobs then"] 
predict=["good how re you you you have any hobbies ?","do you have a job","good how re you you you have any hobbies ?","do you have a job","good how re you you you have any hobbies ?","do you have a job","good how re you you you have any hobbies ?","do you have a job"]
evaluate_predictions(target, predict)
