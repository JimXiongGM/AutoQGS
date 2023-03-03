import os
import subprocess
import tempfile
from typing import Dict, Iterable, List, Set, Tuple, Union

from allennlp.training.metrics.metric import Metric


class Meteor:
    """
    original:
        - https://github.com/Maluuba/nlg-eval/tree/master/nlgeval/pycocoevalcap/meteor
    usage:
        - java -Xmx2G -jar meteor-1.5.jar pred.txt reference.txt -l en -norm
    """

    def __init__(self, meteor_jar="meteor-1.5.jar"):
        self.meteor_jar = (
            meteor_jar
            if meteor_jar
            else os.path.join(os.path.dirname(os.path.abspath(__file__)), "meteor-1.5.jar")
        )
        assert os.path.isfile(
            self.meteor_jar
        ), f"meteor_jar not found. download it from `https://github.com/Maluuba/nlg-eval/blob/master/nlgeval/pycocoevalcap/meteor/meteor-1.5.jar`"

        assert os.path.isfile(
            os.path.join(os.path.dirname(self.meteor_jar), "data/paraphrase-en.gz")
        ), f"paraphrase-en.gz not found, download from `https://github.com/Maluuba/nlg-eval/raw/master/nlgeval/pycocoevalcap/meteor/data/paraphrase-en.gz` and put it in [path of meteor-1.5.jar]/data/paraphrase-en.gz"

        self.meteor_cmd = " ".join(
            [
                "java",
                "-Xmx2G",
                "-jar",
                self.meteor_jar,
                "{pred}",
                "{reference}",
                "-l",
                "en",
                "-norm",
            ]
        )

    def compute_score(self, tgts: List[str], preds: List[str]):
        """calculate meteor score with tmp file.

        Args:
            tgts (List[str]): one sentence per line.
            preds (List[str]): one sentence per line.

        Returns:
            scores: List[float]
            final_score: score
        """
        assert len(tgts) == len(preds), f"len(tgts) {len(tgts)} != len(preds) {len(preds)}"
        assert isinstance(tgts[0], str) and isinstance(
            preds[0], str
        ), f"type err, both must be str. tgts:{type(tgts)} preds: {type(preds)}"

        # Clean up a NamedTemporaryFile on your own
        # delete=True means the file will be deleted on close
        pred_tmp = tempfile.NamedTemporaryFile(mode="w", dir="./", delete=True)
        ref_tmp = tempfile.NamedTemporaryFile(mode="w", dir="./", delete=True)
        for pred, tgt in zip(preds, tgts):
            pred_tmp.write(f"{pred}\n")
            ref_tmp.write(f"{tgt}\n")

        pred_tmp.flush()
        ref_tmp.flush()

        outputs = subprocess.getoutput(self.meteor_cmd.format(pred=pred_tmp.name, reference=ref_tmp.name))

        scores, final_score = self._parse_meteor(outputs=outputs)

        pred_tmp.close()  # deletes the file
        ref_tmp.close()  # deletes the file

        return scores, final_score

    def _parse_meteor(self, outputs):
        outputs = outputs.split("\n")
        scores = [float(line.split()[-1]) for line in outputs if line.startswith("Segment")]
        final_score = float(outputs[-1].split()[-1])
        return scores, final_score


class METEOR(Metric):
    """
    meteor wrapper.
    """

    def __init__(self, exclude_words: Set[int] = None, meteor_jar=None) -> None:
        self._exclude_words = set(exclude_words) if exclude_words else set()
        self.tgts = []
        self.preds = []
        self.meteor = Meteor(meteor_jar)

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
        scores, final_score = self.meteor.compute_score(tgts=self.tgts, preds=self.preds)
        if reset:
            self.reset()
        return {"METEOR": final_score}


if __name__ == "__main__":
    """
    cd metrics
    python meteor.py
    """
    prediction = ["what is the place of death of francis iv duke of modena 's heir ?"]
    references = ["what is the place of death of francis iv duke of modena 's children ?"]
    scorer = METEOR("meteor-1.5.jar")
    scorer(prediction, references)
    score = scorer.get_metric()
    print(score)
