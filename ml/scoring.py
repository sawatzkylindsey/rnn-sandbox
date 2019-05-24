
import math
import pdb


accuracy = lambda xy, result: (True, 1) if xy.y == result.prediction() else (False, 0)


def rank_score_linear(score_threshold=1.0):
    def _fn(xy, result):
        rank = result.rank_of(xy.y, True)
        score = 1.0 - (rank / float(len(result.labels)))
        return True if score >= score_threshold else False, score

    return _fn


def rank_score_exponential(score_threshold=1.0):
    def _fn(xy, result):
        divisor = len(result.labels) / 5.0
        rank = result.rank_of(xy.y, True)
        score = 1.0 / math.exp(rank / divisor)
        return True if score >= score_threshold else False, score

    return _fn


def descrete_rank(top_k=None, top_percent=None):
    assert top_k is None or top_percent is None, "top_k and top_percent are mutually exclusive"

    if top_percent is not None:
        assert top_percent > 0.0 and top_percent <= 1.0, top_percent

    if top_k is not None:
        assert top_k >= 0

    def _fn(xy, result):
        #rank = result.rank_of(xy.y, True)         # Use the insertion-sort method.
        rank = result.rank_of(xy.y, True, top_k) # Use the nlargest method.
        cutoff = int(len(result.labels) * top_percent) if top_k is None else top_k
        return (True, 1) if rank <= cutoff else (False, 0)

    return _fn

