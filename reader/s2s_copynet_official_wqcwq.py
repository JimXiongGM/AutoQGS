import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from allennlp.common.util import END_SYMBOL, START_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField, NamespaceSwappingField, TensorField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from common.common_dataclean import _clean_white_spaces
from common.common_utils import read_json, read_jsonl, replaces
from data_helper import init_nltk
from data_helper.freebase_wrapper import FreebaseWrapper
from utils.helper import replace_mid_in_sparql

logger = logging.getLogger(__name__)

_map = {
    "_": " _ ",
    "^^http://www.w3.org/2001/XMLSchema#dateTime": "",
    "^^http://www.w3.org/2001/XMLSchema#date": "",
    "^^http://www.w3.org/2001/XMLSchema#gYearMonth": "",
    "^^http://www.w3.org/2001/XMLSchema#float": "",
    "^^http://www.w3.org/2001/XMLSchema#gYear": "",
    "^^http://www.w3.org/2001/XMLSchema#integer": "",
}

SPARQL_CLEAN_MAP = {
    "prefix ns: <http://rdf.freebase.com/ns/>": "",
    "filter ( ?x != ?c )": "",
    "filter ( !isliteral ( ?x ) or lang ( ?x ) = '' or langmatches ( lang ( ?x ) , 'en' ) )": "",
    "ns:": "",
}


def _clean_s_expr(s):
    s = replaces(s, _map)
    s = replaces(s, SPARQL_CLEAN_MAP)
    s = _clean_white_spaces(s)
    return s.strip()


class CopyNetDatasetReader(DatasetReader):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs,
        )
        self.fb = FreebaseWrapper(end_point="http://localhost:8890/logical_form")
        self._target_namespace = "tokens"
        self.tokenizer = init_nltk(lower=True)
        self.SOS = START_SYMBOL
        self.EOS = END_SYMBOL

        self._source_token_indexers = {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
        self._target_token_indexers = {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}

    def _read(self, dir_path) -> Iterable[Instance]:
        for data in read_json(dir_path):
            instance = self.text_to_instance(**data)
            yield instance

    @staticmethod
    def _tokens_to_ids(tokens: List[Token]) -> List[int]:
        ids: Dict[str, int] = {}
        out: List[int] = []
        for token in tokens:
            out.append(ids.setdefault(token.text, len(ids)))
        return out

    def text_to_instance(self, ID, question, sparql_nomid, **kwargs) -> Instance:  # type: ignore
        """
        sparql_nomid or s_expr_nomid
        """
        # s_expr_nomid=s_expr_nomid.replace(")"," )")
        logical_form = _clean_s_expr(sparql_nomid.lower())
        if "order by" in logical_form:
            _index = logical_form.index("order by")
            logical_form = logical_form[:_index].strip()

        # print(logical_form)
        # logical_form = (
        #     replace_mid_in_sparql(logical_form=logical_form, fb=self.fb)
        #     .lower()
        #     .replace("prefix ns: <http://rdf.freebase.com/ns/> ", "")
        # )
        tokenized_source = [Token(i) for i in self.tokenizer(logical_form)]
        source_field = TextField(tokenized_source)

        # For each token in the source sentence, we keep track of the matching token
        # in the target sentence (which will be the OOV symbol if there is no match).
        source_to_target_field = NamespaceSwappingField(tokenized_source, self._target_namespace)

        meta_fields = {}
        meta_fields["ID"] = ID
        meta_fields["logical_form"] = logical_form
        meta_fields["comp_type"] = kwargs.get("compositionality_type", "")
        meta_fields["source_tokens"] = tokenized_source

        fields_dict = {
            "source_tokens": source_field,
            "source_to_target": source_to_target_field,
        }

        if question is not None:
            ques_tokens = self.tokenizer(question)
            meta_fields["ques_tokens"] = ques_tokens

            tokenized_target = [Token(i) for i in ques_tokens]
            tokenized_target.insert(0, Token(self.SOS))
            tokenized_target.append(Token(self.EOS))
            target_field = TextField(tokenized_target)

            fields_dict["target_tokens"] = target_field
            source_and_target_token_ids = self._tokens_to_ids(tokenized_source + tokenized_target)
            source_token_ids = source_and_target_token_ids[: len(tokenized_source)]
            fields_dict["source_token_ids"] = TensorField(torch.tensor(source_token_ids))
            target_token_ids = source_and_target_token_ids[len(tokenized_source) :]
            fields_dict["target_token_ids"] = TensorField(torch.tensor(target_token_ids))
        else:
            source_token_ids = self._tokens_to_ids(tokenized_source)
            fields_dict["source_token_ids"] = TensorField(torch.tensor(source_token_ids))

        fields_dict["metadata"] = MetadataField(meta_fields)
        return Instance(fields_dict)

    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["source_tokens"]._token_indexers = self._source_token_indexers  # type: ignore
        if "target_tokens" in instance.fields:
            instance.fields["target_tokens"]._token_indexers = self._target_token_indexers  # type: ignore


if __name__ == "__main__":
    reader = CopyNetDatasetReader()
    for ins in reader._read("datasets/GrailQA/Generalization/graph_to_seq/dev.json"):
        _s = ins
