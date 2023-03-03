import logging
from typing import Dict, Iterable, Optional

import torch
from allennlp.data import DatasetReader, Field, Instance
from allennlp.data.fields import MetadataField, TensorField, TextField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer, Token
from tqdm import tqdm

from common.common_utils import colorful, read_json

logger = logging.getLogger(__name__)


class RerankDatasetReader(DatasetReader):
    def __init__(
        self,
        model_name,
        tokenizer_kwargs=None,
        max_instances_percent=None,
        skip_visited=False,
        **kwargs,
    ):
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs,
        )

        max_length = 512
        self.max_instances_percent = max_instances_percent
        self.tokenizer = PretrainedTransformerTokenizer(
            model_name,
            max_length=max_length,
            add_special_tokens=True,
            tokenizer_kwargs=tokenizer_kwargs,
        )
        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(model_name, tokenizer_kwargs=tokenizer_kwargs)
        }
        self.skip_visited = skip_visited

    def _read(self, file_path, total=None) -> Iterable[Instance]:
        _total = None
        datas = read_json(file_path)[:total]
        _iter = tqdm(datas, ncols=100, desc="dataset loading")
        if self.max_instances_percent:
            _total = max(self.max_instances_percent * len(datas), 3)

        if self.skip_visited:
            visited = set()

        _c = 0
        for idx, item in enumerate(self.shard_iterable(_iter)):
            # debug
            # if idx < 2390:
            #     continue
            if _total and idx >= _total:
                continue

            # if float(item["bleu-4"]) > 0.0:

            pred_ques = " ".join(item["predicted_tokens"])
            # if self.skip_visited:
            #     if pred_ques in visited:
            #         continue
            #     else:
            #         visited.add(pred_ques)

            tmp = {}
            tmp["ID"] = item["ID"]
            tmp["passage"] = item["input"]
            tmp["pred_ques"] = pred_ques
            tmp["score"] = float(item["bleu-4"])
            tmp["golden_ques"] = item["output"]
            tmp["compositionality_type"] = item["compositionality_type"]

            tmp["bleu-4"] = item["bleu-4"]
            if "beam_logit" not in item:
                _c += 1
                print(colorful(f"err: {_c}"))
                continue
            tmp["beam_logit"] = item["beam_logit"]

            yield self.text_to_instance(**tmp)

    def text_to_instance(  # type: ignore
        self,
        ID,
        passage: str,
        pred_ques: str,
        score=None,
        **kwargs,
    ) -> Instance:

        fields_dict = {}
        meta_fields = {}
        meta_fields["ID"] = ID
        meta_fields["pred_ques"] = pred_ques

        _input = f"{passage} [sep] {pred_ques}".lower()
        tokenized_source = self.tokenizer.tokenize(_input)
        text_field = TextField(tokenized_source)
        fields_dict["tokens"] = text_field

        for n in ["golden_ques", "bleu-4", "beam_logit", "compositionality_type"]:
            _t = kwargs.get(n, None)
            if _t:
                if isinstance(_t, dict):
                    meta_fields[n] = list(_t.items())
                else:
                    meta_fields[n] = _t

        if score is not None:
            score_field = TensorField(torch.tensor(score))
            fields_dict["score"] = score_field

        fields_dict["metadata"] = MetadataField(meta_fields)
        return Instance(fields_dict)

    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["tokens"]._token_indexers = self._token_indexers  # type: ignore


if __name__ == "__main__":
    reader = RerankDatasetReader(model_name="pretrained_models/bert-base-uncased", max_instances=100)

    for ins in reader._read("datasets/for_rerank/scores-50-train-10%.json"):
        _s = ins
