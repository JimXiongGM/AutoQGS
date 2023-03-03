from typing import Dict, Iterable, List, Set, Tuple, Union

from allennlp.training.metrics.metric import Metric
from rouge import Rouge


def cal_Rouge():
    """
    usage:
        rouge = cal_Rouge()
        rouge(golden=self.tgts, pred=self.preds, avg=True)

    Returns:
        dict
    """
    rouge = Rouge()

    def _cal_rouge(golden: Union[str, List[str]], pred: Union[str, List[str]], avg: bool = True):
        score = rouge.get_scores(hyps=pred, refs=golden, avg=avg)
        res = {}
        res["ROUGE-1-P"] = score["rouge-1"]["p"]
        res["ROUGE-1-R"] = score["rouge-1"]["r"]
        res["ROUGE-1-F"] = score["rouge-1"]["f"]

        res["ROUGE-2-P"] = score["rouge-2"]["p"]
        res["ROUGE-2-R"] = score["rouge-2"]["r"]
        res["ROUGE-2-F"] = score["rouge-2"]["f"]

        res["ROUGE-L-P"] = score["rouge-l"]["p"]
        res["ROUGE-L-R"] = score["rouge-l"]["r"]
        res["ROUGE-L-F"] = score["rouge-l"]["f"]
        return res

    return _cal_rouge


class pyROUGE(Metric):
    """
    ROUGE
    """

    def __init__(
        self,
        exclude_words: Set[int] = None,
    ) -> None:
        self._exclude_words = set(exclude_words) or set()
        self.tgts = []
        self.preds = []
        self.rouge = cal_Rouge()

    def reset(self) -> None:
        self.tgts = []
        self.preds = []

    def _filter_exclude(self, words):
        words = [i for i in words if i not in self._exclude_words]
        return words

    def __call__(
        self,  # type: ignore
        predictions: Union[List[str], List[List[str]]],
        gold_targets: Union[List[str], List[List[str]]],
    ) -> None:
        """[summary]

        Args:
            gold_targets (List[str]): [description]
        """
        assert len(predictions) == len(
            gold_targets
        ), f"len(predictions) {len(predictions)} != len(gold_targets) {len(gold_targets)} !"
        if not predictions or not gold_targets:
            return

        if isinstance(predictions[0], list):
            predictions = [self._filter_exclude(i) for i in predictions]
            predictions = [" ".join(i) for i in predictions]

        if isinstance(gold_targets[0], list):
            gold_targets = [self._filter_exclude(i) for i in gold_targets]
            gold_targets = [" ".join(i) for i in gold_targets]

        self.tgts.extend(gold_targets)
        self.preds.extend(predictions)

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        metric = self.rouge(golden=self.tgts, pred=self.preds, avg=True)
        if reset:
            self.reset()
        return metric
