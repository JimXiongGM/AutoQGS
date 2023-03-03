import json
import os


def Jaccard_coefficient(A, B):
    if not isinstance(A, set):
        A = set(A)
    if not isinstance(B, set):
        B = set(B)
    nominator = len(A & B)
    denominator = len(A | B)
    if denominator == 0:
        return 0
    else:
        return nominator / denominator


def order_by_jaccard(query, candicates, reverse=True):
    """
    input:
        query: list
        candicates: list of list
    return:
        [(index, score) ... ]
    """
    scores = [(idx, Jaccard_coefficient(query, c)) for idx, c in enumerate(candicates)]
    scores.sort(key=lambda x: x[1], reverse=reverse)
    return scores


def topK_lists(*lists, topK=5, reverse=True):
    """
    usage:
    x1 = [0.9, 0.7, 0.2]
    x2 = [0.5, 0.1]
    topK_lists(x1, x2, x1)
    """
    if not lists:
        return []
    tmp_topk = [(v, [i]) for i, v in enumerate(lists[0])]
    if len(lists) == 1:
        return tmp_topk[:topK]
    tmp_topk = sorted(tmp_topk, key=lambda x: x[0], reverse=reverse)[:topK]
    for i in range(1, len(lists)):
        x = lists[i]
        # (score)
        tmp_topk = sorted(
            [(v0[0] * v, v0[1] + [i]) for v0 in tmp_topk for i, v in enumerate(x)],
            key=lambda x: x[0],
            reverse=reverse,
        )[:topK]
    return tmp_topk  # [(score, [1, 2]), ...]


