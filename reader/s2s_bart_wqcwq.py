import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer, Token

from common.common_utils import read_json
from data_helper import init_nltk

nltk_tokenizer = init_nltk(lower=True)
logger = logging.getLogger(__name__)


# 废弃
class BARTDatasetReader(DatasetReader):
    def __init__(
        self,
        model_name,
        tokenizer_kwargs=None,
        setting=None,
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs,
        )
        self.setting = setting
        assert setting in [
            "sparql-nomid",
            "human-NL-nodesc",
            # "only-autoprompt",
            # "sparql-autoprompt",
            # "middesc-autoprompt",
            # "sparql-middesc-autoprompt",
        ]
        self.tokenizer = PretrainedTransformerTokenizer(
            model_name,
            max_length=1024,
            add_special_tokens=True,
            tokenizer_kwargs=tokenizer_kwargs,
        )
        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(model_name, tokenizer_kwargs=tokenizer_kwargs)
        }

    def _read(self, dir_path) -> Iterable[Instance]:

        for data in read_json(dir_path):
            if self.src_key not in data.keys():
                # data[self.src_key] = data["sparql_nomid"]
                continue
            instance = self.text_to_instance(**data)
            yield instance

    def text_to_instance(
        self,
        ID,
        question,
        **kwargs,
    ) -> Instance:  # type: ignore
        _input = kwargs[self.src_key]
        if isinstance(_input, list):
            _input = " ".join(_input)
        _input = _input.lower()
        tokenized_source = self.tokenizer.tokenize(_input)
        source_field = TextField(tokenized_source)

        meta_fields = {}
        meta_fields["ID"] = ID
        meta_fields["_input"] = _input
        meta_fields["compositionality_type"] = kwargs.get("compositionality_type", "")
        meta_fields["source_tokens"] = tokenized_source

        fields_dict = {
            "source_tokens": source_field,
        }

        if question is not None:
            question = question.lower()
            meta_fields["question"] = question
            meta_fields["ques_tokens"] = nltk_tokenizer(text=question)
            tokenized_target = self.tokenizer.tokenize(question)
            fields_dict["target_tokens"] = TextField(tokenized_target)

        fields_dict["metadata"] = MetadataField(meta_fields)

        return Instance(fields_dict)

    def apply_token_indexers(self, instance: Instance) -> None:
        # type: ignore
        instance.fields["source_tokens"]._token_indexers = self._token_indexers
        if "target_tokens" in instance.fields:
            # type: ignore
            instance.fields["target_tokens"]._token_indexers = self._token_indexers


if __name__ == "__main__":
    reader = BARTDatasetReader(model_name="pretrained_models/facebook/bart-base")
    for ins in reader._read("preprocess/wqcwq_OOV_v1.0/test.json"):
        _s = ins
