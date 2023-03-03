import numpy as np


def cal_acc(preds, goldens):
    assert len(preds) == len(goldens), f"{len(preds)} is not equal to {len(goldens)}"
    correct = 0.0
    for x, y in zip(preds, goldens):
        if x == y:
            correct += 1
    return correct / (len(goldens) + 1e-9)


def cal_precision(preds, goldens):
    tp = len(set(preds) & set(goldens))
    return tp / (len(preds) + 1e-9)


def cal_recall(preds, goldens):
    tp = len(set(preds) & set(goldens))
    return tp / (len(goldens) + 1e-9)


def cal_PRF1(preds, goldens):
    precision = cal_precision(preds, goldens)
    recall = cal_recall(preds, goldens)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    return precision, recall, f1


def cal_PRF1_average(prf1s):
    ave_pre = sum([i[0] for i in prf1s]) / len(prf1s)
    ave_rec = sum([i[1] for i in prf1s]) / len(prf1s)
    ave_f1 = sum([i[2] for i in prf1s]) / len(prf1s)
    return (
        ave_pre,
        ave_rec,
        ave_f1,
    )


def statistic_info(nums=[1, 1, 2, 3]):
    from scipy import stats

    nums = np.array(nums)
    _max = nums.max()
    _min = nums.min()
    _mean = nums.mean()
    _median = np.median(nums)
    _most = stats.mode(nums)[0][0]
    _percent_25 = np.percentile(nums, 25)
    _percent_75 = np.percentile(nums, 75)
    _percent_90 = np.percentile(nums, 90)
    _percent_95 = np.percentile(nums, 95)
    _std = np.std(nums)
    _skew = stats.skew(nums)  # skewness
    _kurtosis = stats.kurtosis(nums)  # kurtosis

    return {
        "mean": _mean,
        "max": _max,
        "min": _min,
        "mode": _most,
        "median": _median,
        "25% percentile": _percent_25,
        "75% percentile": _percent_75,
        "90% percentile": _percent_90,
        "95% percentile": _percent_95,
        "standard deviation": _std,
        "skewness": _skew,
        "kurtosis": _kurtosis,
    }
