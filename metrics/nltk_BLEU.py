from typing import Dict, Iterable, List, Set, Tuple, Union

from allennlp.training.metrics.metric import Metric
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def cal_BLEU():
    smooth = SmoothingFunction()
    ngram_dict = {
        1: (1,),
        2: (1 / 2, 1 / 2),
        3: (1 / 3, 1 / 3, 1 / 3),
        4: (1 / 4, 1 / 4, 1 / 4, 1 / 4),
    }

    def _cal_bleu(golden: List[str], pred: List[str], ngram=4):
        weights = ngram_dict[int(ngram)]
        try:
            score = sentence_bleu(
                golden,
                pred,
                weights=weights,
                smoothing_function=smooth.method4,
            )
        except:
            score = 0
        return score

    return _cal_bleu


class NLTKBLEU(Metric):
    """
    Bilingual Evaluation Understudy (BLEU).
    """

    def __init__(
        self,
        exclude_words: Set[int] = None,
        tokenizer=lambda x: x.split(),
        ngrams=[1, 2, 3, 4],
    ) -> None:
        self._exclude_words = set(exclude_words) if exclude_words else set()
        self.tgts = []
        self.preds = []
        self.bleu = cal_BLEU()
        self.tokenizer = tokenizer
        self.ngrams = ngrams

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

        if isinstance(predictions[0], str):
            predictions = [self.tokenizer(i) for i in predictions]
        predictions = [self._filter_exclude(i) for i in predictions]

        if isinstance(gold_targets[0], str):
            gold_targets = [self.tokenizer(i) for i in gold_targets]
        gold_targets = [self._filter_exclude(i) for i in gold_targets]

        self.tgts.extend(gold_targets)
        self.preds.extend(predictions)

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        metric = {}
        for n in self.ngrams:
            avg = sum(
                [self.bleu([golden], pred, ngram=n) for golden, pred in zip(self.tgts, self.preds)]
            ) / len(self.preds)
            metric[f"BLEU-{n}"] = avg
        if reset:
            self.reset()
        return metric
