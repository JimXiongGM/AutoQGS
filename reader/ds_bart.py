import logging
import os
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField, TensorField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

from common.common_dataclean import _clean_white_spaces
from common.common_utils import colorful, jsonl_generator, read_json, read_jsonl, replaces, save_to_json

os.environ["TOKENIZERS_PARALLELISM"] = "false"


logger = logging.getLogger(__name__)

"""
Given CVT, generate NL
"""

SPARQL_CLEAN_MAP = {
    "prefix ns: <http://rdf.freebase.com/ns/>": "",
    "filter ( ?x != ?c )": "",
    "filter ( !isliteral ( ?x ) or lang ( ?x ) = '' or langmatches ( lang ( ?x ) , 'en' ) )": "",
}

VAR_MAP = {
    "?y": "entity y",
    "?k": "entity k",
    "?c": "entity c",
}
SPECIAL1 = [" is a ", " is an ", " was a ", " was an ", " is the ", " was the "] + [
    " describes ",
    " served as ",
    " has been ",
]
SPECIAL2 = [" is ", " was ", " were ", " are "]

# preds with tail entity: start and end with <>; not start with `m.` and `g.`
TYPE_PREDs = set(
    [
        "type.object.type",
        "kg.object_profile.prominent_type",
        "topic_server.webref_cluster_members_type",
        "type.property.expected_type",
        "topic_server.schemastaging_corresponding_entities_type",
        "type.property.schema",
        "type.property.reverse_property",
    ]
)


# @DatasetReader.register("DSDatasetReader")
class DSDatasetReader(DatasetReader):
    def __init__(
        self,
        model_name,
        tokenizer_kwargs=None,
        setting=None,
        max_instances_percent=1,
        top_beam=None,
        decoder_instructions=False,
        beam_sample="all",
        beam_score_map=None,
        reward_map=None,
        ids_map=None,
        mode="test",
        only_constrain=None,
        max_length=512,
        aug_mode=None,
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs,
        )
        self.reward_map = reward_map
        self.max_instances = kwargs.get("max_instances", None)
        self.beam_sample = beam_sample
        self.random_seed = 123
        self.max_instances_percent = max_instances_percent or 1
        self.top_beam = top_beam
        self.decoder_instructions = decoder_instructions
        # self.rerun_ids = set(read_json(rerun_ids)) if rerun_ids else None
        self.mode = mode
        self.aug_mode = aug_mode
        if mode != "train":
            self.max_instances_percent = 1

        # no need for auto-prompt
        if ids_map:
            _ids_map = read_json(ids_map)
            if "test-unseen" in _ids_map:
                _ids_map["test"] = _ids_map["test-iid"] + _ids_map["test-unseen"]
                self.test_iid_id = set(_ids_map["test-iid"])
                self.test_unseen_id = set(_ids_map["test-unseen"])
            else:
                self.test_iid_id = None
            self.ids = _ids_map[mode] if ids_map else None

        if mode == "train" and aug_mode:
            self.ids += [
                i["ID"] for i in read_jsonl("datasets/wqcwq-aug/augtrain-1_times.jsonl") if "@AUG@" in i["ID"]
            ]

        if only_constrain is not None:
            self.constrain_ids = set(read_json("datasets/wqcwq-all-infos-v1.0/constrains_ids.json"))
            self.max_instances_percent = 1.0
            if only_constrain:
                self.ids = [i for i in self.ids if i in self.constrain_ids]
            else:
                self.ids = [i for i in self.ids if i not in self.constrain_ids]

        self.setting = setting
        assert setting in [
            "ds-merge-desc-realent",
            "ds-merge-desc-pattent",
            "sparql-nomid",
            "human-NL-nodesc",
            "only-autoprompt",
            "sparql-autoprompt",
            "sparql-middesc",
        ]

        if setting[:3] in ["ds-"]:
            # max_length = 512
            self.ids = None

        print(colorful(f"DatasetReader setting: {setting}\tmode: {mode}"))
        if self.ids:
            self.ids = set(self.ids[: int(len(self.ids) * self.max_instances_percent)])
            print(
                colorful(
                    f"raw data len: {len(self.ids)}\t now percent: {self.max_instances_percent} now data len: {len(self.ids)}"
                )
            )

        self.tokenizer = PretrainedTransformerTokenizer(
            model_name,
            max_length=max_length,
            add_special_tokens=True,
            tokenizer_kwargs=tokenizer_kwargs,
        )
        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(model_name, tokenizer_kwargs=tokenizer_kwargs)
        }
        if self.beam_sample == "jaccard":
            self.jaccard_map = read_jsonl(beam_score_map)
            self.jaccard_map = {i["ID"]: i["beams_simi_jaccard_sparql-desc"] for i in self.jaccard_map}

        if self.reward_map:
            self.default_reward = 1.0
            if self.mode == "train":
                tmp = read_jsonl(reward_map, desc=f"Loading {reward_map}")
                self._reward_map = {i["ID"]: i["bleu_4"] for i in tmp}
            else:
                self._reward_map = {}

    def _read(self, dir_path) -> Iterable[Instance]:
        self.skips = {}

        if self.mode == "train" and self.max_instances_percent < 0.01:
            r = 2
        else:
            r = 1

        for _ in range(r):
            _iter = jsonl_generator(
                dir_path,
                colour="green",
                disable=False,
            )
            for idx, item in enumerate(self.shard_iterable(_iter)):
                # debug
                # if item["ID"] in skips:
                #     continue
                for i in self._parse_item(item):
                    if i is not None:
                        yield i

    def _parse_item(self, item):
        # ds: unified processing for cvt pred
        if self.setting[:3] == "ds-" and self.mode == "train":
            # Instantiate all entities, no placeholders
            if self.setting == "ds-merge-desc-realent":
                # cvt
                _input = ""
                if item["ID"].startswith("cvt"):
                    # input: <P> <T> <D>
                    # Details:
                    # - Preserve case
                    # - Remove mids corresponding to predicates has_no_value
                    # - Keep predicates if they are not m.g.
                    tmp_list = []
                    for i in item["in"]:
                        mid, name, pred = i
                        pred = "R@" + pred
                        tmp_list.append([pred, mid, name])
                    for i in item["out"]:
                        pred, mid, name = i
                        tmp_list.append([pred, mid, name])

                    # input
                    bad_mids = set()
                    for line in tmp_list:
                        pred, mid, name = line
                        if (
                            pred.endswith("has_no_value")
                            or pred.endswith("has_value")
                            or name.lower() in ["from", "to"]
                        ):
                            bad_mids.add(mid)
                            continue

                        if mid in item["mid_info"]:
                            desc = item["mid_info"][mid]["desc"]
                        else:
                            desc = "No description."

                        _input += f" <P> {pred} <T> {name} <D> {desc}"

                    # output
                    _map = {
                        " "
                        + item["mid_info"][k]["name"]
                        + " ": " <E> "
                        + item["mid_info"][k]["name"]
                        + " </E> "
                        for k in item["mid_info"].keys()
                        if k not in bad_mids
                    }
                    _output = replaces(text=" " + item["content"] + " ", maps=_map, ignorecase=True)

                # pred
                elif item["ID"].startswith("pred"):
                    # input: <H> <D> <P> <T> <D>
                    e0name = item["e0name"]
                    e1name = item["e1name"]
                    e0desc = item["e0desc"] if item["e0desc"] else "No description."
                    e1desc = item["e1desc"] if item["e1desc"] else "No description."
                    _input = f"<H> {e0name} <D> {e0desc} <P> {item['pred']} <T> {e1name} <D> {e1desc}"

                    # output
                    _map = {
                        f" {e0name} ": f" <E> {e0name} </E> ",
                        f" {e1name} ": f" <E> {e1name} </E> ",
                    }
                    _output = replaces(text=" " + item["output"] + " ", maps=_map, ignorecase=True)

                else:
                    raise ValueError(f"error format.")

                item["input"] = _input.strip()  # .lower()
                item["output"] = _output.strip()  # .lower()

            # Using placeholders with type placeholders
            # pred
            # [person] [Bill Gates' description] [person.person.wife] [wife] [Melinda's description]
            # cvt
            # [film.performence.actor] [actor] [Actor A's description]
            # [film.performence.actor] [actor] [Actor B's description]
            # [film.performence.director] [director] [Director's description]
            elif self.setting == "ds-merge-desc-pattent":
                # cvt
                _input = ""
                if item["ID"].startswith("cvt"):
                    tmp_list = []
                    for i in item["in"]:
                        mid, name, pred = i
                        pred = "R@" + pred
                        tmp_list.append([pred, mid, name])
                    for i in item["out"]:
                        pred, mid, name = i
                        tmp_list.append([pred, mid, name])

                    # input
                    valid_spos = []
                    for line in tmp_list:
                        pred, mid, name = line
                        if (
                            pred.endswith("has_no_value")
                            or pred.endswith("has_value")
                            or name.lower() in ["from", "to"]
                            or name.lower()[:2] in ["m.", "g."]
                        ):
                            continue
                        valid_spos.append(line)

                    patt_maps = defaultdict(list)
                    for line in valid_spos:
                        pred, mid, name = line
                        if pred not in TYPE_PREDs:
                            if mid in item["mid_info"]:
                                desc = item["mid_info"][mid]["desc"]
                            else:
                                desc = "No description."

                            patt = pred.split(".")[-2] if pred.startswith("R@") else pred.split(".")[-1]
                            patt = f"[{patt}]"
                            patt_maps[patt].append(name)

                            desc = DSDatasetReader._replace_ent_in_desc(name, patt, desc)

                            _input += f" <P> {pred} <T> {patt} <D> {desc}"

                    item["patt_maps"] = patt_maps

                    # output
                    _map = {f" {name} ": f" {patt} " for patt, names in patt_maps.items() for name in names}
                    _o = (" " + item["content"] + " ").replace("[", "").replace("]", "")
                    _output = replaces(text=_o, maps=_map, ignorecase=True)

                # pred
                elif item["ID"].startswith("pred"):
                    e0name = item["e0name"]
                    e1name = item["e1name"]
                    pred = item["pred"]
                    h_type, t_type = pred.split(".")[-2:]
                    h_type, t_type = f"[{h_type}]", f"[{t_type}]"

                    e0desc = item["e0desc"] if item["e0desc"] else "No description."
                    e0desc = DSDatasetReader._replace_ent_in_desc(e0name, h_type, e0desc)

                    e1desc = item["e1desc"] if item["e1desc"] else "No description."
                    e1desc = DSDatasetReader._replace_ent_in_desc(e1name, t_type, e1desc)

                    _input = f"<H> {h_type} <D> {e0desc} <P> {pred} <T> {t_type} <D> {e1desc}"

                    # output
                    _map = {
                        f" {e0name} ": f" {h_type} ",
                        f" {e1name} ": f" {t_type} ",
                    }
                    _o = (" " + item["output"] + " ").replace("[", "").replace("]", "")
                    _output = replaces(text=_o, maps=_map, ignorecase=True)

                else:
                    raise ValueError(f"error format.")

                item["input"] = _input.strip()
                item["output"] = _output.strip()
                item["map"] = _map

                # debug
                if "]n" in item["output"]:
                    x = 1

            else:
                # print(item)
                raise ValueError(f"err setting: {self.setting}")

            # end
            instance = self.text_to_instance(**item)
            yield instance

        # dataset inference
        elif self.setting[:3] == "ds-" and not self.mode == "train":
            if self.setting == "ds-merge-desc-realent":
                # cvt
                for k, info in item["cvt_spos"].items():
                    # input: <P> <T> <D>
                    tmp_list = []
                    for i in info["in"]:
                        mid, name, pred = i
                        pred = "R@" + pred
                        tmp_list.append([pred, mid, name])
                    for i in info["out"]:
                        pred, mid, name = i
                        tmp_list.append([pred, mid, name])

                    # input
                    bad_mids = set()
                    _input = ""
                    for line in tmp_list:
                        pred, mid, name = line
                        if (
                            pred.endswith("has_no_value")
                            or pred.endswith("has_value")
                            or name.lower() in ["from", "to"]
                        ):
                            bad_mids.add(mid)
                            continue

                        if mid in item["mid_info"]:
                            desc = item["mid_info"][mid]["desc"]
                        else:
                            desc = "No description."

                        if name[0] == "?":
                            name = item["var_res_one"][name]

                        name = name.split("^^<http:")[0]
                        _input += f" <P> {pred} <T> {name} <D> {desc}"

                    tmp = {"ID": item["ID"] + "@cvt@" + k, "input": _input.strip()}
                    if tmp["ID"] in self.skips:
                        yield None
                    else:
                        instance = self.text_to_instance(**tmp)
                        yield instance

                # only-pred
                for idx, spo in enumerate(item["pred_spos"]):
                    e0, p, e1 = spo
                    item["mid_info"].setdefault(
                        e0,
                        {
                            "name": item["var_res_one"].get(e0, e0),
                            "desc": "No description.",
                        },
                    )
                    item["mid_info"].setdefault(
                        e1,
                        {
                            "name": item["var_res_one"].get(e1, e1),
                            "desc": "No description.",
                        },
                    )
                    e0name = item["mid_info"][e0]["name"] if e0 in item["mid_info"] else e0
                    e1name = item["mid_info"][e1]["name"] if e1 in item["mid_info"] else e1
                    e0desc = item["mid_info"][e0]["desc"]
                    e1desc = item["mid_info"][e1]["desc"]

                    e0name = e0name.split("^^<http:")[0]
                    e1name = e1name.split("^^<http:")[0]
                    _input = f"<H> {e0name} <D> {e0desc} <P> {p} <T> {e1name} <D> {e1desc}"
                    if e0name[0] == "?" or e1name[0] == "?":
                        continue
                    tmp = {"ID": item["ID"] + f"@pred@{idx}", "input": _input.strip()}
                    if tmp["ID"] in self.skips:
                        yield None
                    else:
                        instance = self.text_to_instance(**tmp)
                        yield instance

            # Has placeholder, using type placeholder
            elif self.setting == "ds-merge-desc-pattent":
                # cvt
                for k, info in item["cvt_spos"].items():
                    _id = item["ID"] + f"@cvt@{k}"
                    _input = ""
                    tmp_list = []
                    for i in info["in"]:
                        mid, name, pred = i
                        pred = "R@" + pred
                        tmp_list.append([pred, mid, name])
                    for i in info["out"]:
                        pred, mid, name = i
                        tmp_list.append([pred, mid, name])

                    # input
                    valid_spos = []
                    for line in tmp_list:
                        pred, mid, name = line
                        if (
                            pred.endswith("has_no_value")
                            or pred.endswith("has_value")
                            or name.lower() in ["from", "to"]
                            or name.lower()[:2] in ["m.", "g."]
                        ):
                            continue
                        valid_spos.append(line)

                    # one-to-many, same predicate can have multiple mappings
                    patt_maps = defaultdict(list)
                    for line in valid_spos:
                        pred, mid, name = line
                        if pred not in TYPE_PREDs:
                            if mid in item["mid_info"]:
                                desc = item["mid_info"][mid]["desc"]
                            else:
                                desc = "No description."

                            patt = pred.split(".")[-2] if pred.startswith("R@") else pred.split(".")[-1]
                            patt = f"[{patt}]"
                            patt_maps[patt].append(name)

                            desc = DSDatasetReader._replace_ent_in_desc(name, patt, desc)
                            desc = desc if desc else "No description."

                            _input += f" <P> {pred} <T> {patt} <D> {desc}"

                    item["patt_maps"] = patt_maps

                    tmp = {
                        "ID": _id,
                        "input": _input.strip(),
                        "patt_maps": patt_maps,
                    }
                    instance = self.text_to_instance(**tmp)
                    yield instance

                # only-pred
                for idx, spo in enumerate(item["pred_spos"]):
                    _id = item["ID"] + f"@pred@{idx}"
                    e0, p, e1 = spo
                    item["mid_info"].setdefault(e0, {"name": "No name.", "desc": "No description."})
                    item["mid_info"].setdefault(e1, {"name": "No name.", "desc": "No description."})
                    e0name = item["mid_info"][e0]["name"] if e0[:2] in ["m.", "g."] else e0
                    e1name = item["mid_info"][e1]["name"] if e1[:2] in ["m.", "g."] else e1

                    h_type, t_type = p.split(".")[-2:]
                    h_type, t_type = f"[{h_type}]", f"[{t_type}]"

                    e0desc = item["mid_info"][e0]["desc"]
                    e0desc = DSDatasetReader._replace_ent_in_desc(e0name, h_type, e0desc)

                    e1desc = item["mid_info"][e1]["desc"]
                    e1desc = DSDatasetReader._replace_ent_in_desc(e1name, t_type, e1desc)

                    _input = f"<H> {h_type} <D> {e0desc} <P> {p} <T> {t_type} <D> {e1desc}"

                    patt_maps = {h_type: e0name, t_type: e1name}

                    tmp = {
                        "ID": _id,
                        "input": _input.strip(),
                        "patt_maps": patt_maps,
                    }
                    instance = self.text_to_instance(**tmp)
                    yield instance

            else:
                # print(item)
                raise ValueError(f"err setting: {self.setting}")

        # no beams
        elif self.setting in ["sparql-nomid", "human-NL-nodesc", "sparql-middesc"]:
            if self.ids is not None and item["ID"] not in self.ids:
                yield None
            else:
                tmp = {}
                tmp["ID"] = item["ID"]
                tmp["output"] = item["question"].lower() if "question" in item else None

                _input = ""
                if "sparql" in self.setting:
                    _input = replaces(item["sparql_nomid"].lower(), SPARQL_CLEAN_MAP)

                # contains all variables and mid's desc
                if "middesc" in self.setting:
                    _input += " descriptions: "
                    for k, v in item["mid_info"].items():
                        n = v["name"]
                        d = v["desc"]
                        if k[0] == "?":
                            _input += f"{n} (the {k}), {d}"
                        else:
                            _input += d

                if "human-NL" in self.setting:
                    # Note! There is question:
                    _patt = item["sparql_NL-nodesc"][-1]
                    _inputs = item["sparql_NL-nodesc"][:-1] + item["patts_number"] + [_patt]
                    _input = " ".join(_inputs)

                tmp["input"] = _input.lower()  # .replace(" variable ", " entity ")
                if "level" in item:
                    tmp["level"] = item["level"]
                yield self.text_to_instance(**tmp)

        # final gen question
        elif self.setting in [
            "only-autoprompt",
            "sparql-autoprompt",
        ]:
            if self.ids is not None and item["ID"] not in self.ids:
                yield None
            else:
                if self.beam_sample == "top1":
                    _beams = item["final_beams"][:1]
                elif self.beam_sample == "random":
                    random.seed(self.random_seed)
                    _beam = random.choice(item["final_beams"])
                    _beams = [_beam]
                elif self.beam_sample == "jaccard":
                    simis = self.jaccard_map[item["ID"]]
                    simis = sorted(
                        [(idx, s) for idx, s in enumerate(simis)],
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    _i, jaccard_score = simis[0]
                    _beams = [item["final_beams"][_i]]
                    item["jaccard_score"] = jaccard_score
                elif self.beam_sample == "all":
                    if not self.top_beam:
                        print(colorful(f"USE beams top {self.top_beam} !!", color="red"))
                    _beams = item["final_beams"][: self.top_beam]
                else:
                    raise ValueError(f"err beam_sample: {self.beam_sample}")

                for i, _beam in enumerate(_beams):
                    logit, _auto_pmt = _beam
                    _input = ""

                    if "sparql" in self.setting:
                        _input += replaces(item["sparql_nomid"], SPARQL_CLEAN_MAP)

                    if not _input.strip().endswith("."):
                        _input += "."

                    _input += " schema description: "
                    _input += _auto_pmt

                    if self.decoder_instructions:
                        _output = "question: " + item["question"]
                    else:
                        if not _input.strip().endswith("."):
                            _input += "."
                        _input += " question:"
                        _output = item["question"]

                    tmp = {}
                    tmp["ID"] = item["ID"] + f"@beam{i}"
                    tmp["input"] = _input.lower()
                    tmp["output"] = _output.lower()
                    tmp["beam_logit"] = logit

                    if "compositionality_type" in item:
                        tmp["compositionality_type"] = item["compositionality_type"]

                    if "jaccard_score" in item:
                        tmp["jaccard_score"] = jaccard_score

                    if "level" in item:
                        tmp["level"] = item["level"]

                    if self.reward_map:
                        tmp["reward"] = (
                            float(self._reward_map[item["ID"]][i]) if item["ID"] in self._reward_map else 1.0
                        )

                    yield self.text_to_instance(**tmp)

        else:
            raise ValueError(f"err setting: {self.setting}")

    @staticmethod
    def _replace_ent_in_desc(name, patt, desc, strict=False):
        """
        Try to replace ent in desc
        name: str. Harry
        patt: str. [people]
        strict = false when DS. true when predict
        """
        if not desc or desc == "No description.":
            pass

        elif name.lower() in desc.lower():
            desc = replaces(text=f" {desc} ", maps={f" {name} ": f" {patt} "}, ignorecase=True)

        # Note! When encountering this, replace the previous part directly
        elif any([True if i in desc else False for i in SPECIAL1]):
            for i in SPECIAL1:
                if i in desc:
                    desc = patt + desc[desc.find(i) :]
                    break

        elif any([True if i in desc else False for i in SPECIAL2]):
            for i in SPECIAL2:
                if i in desc[:50]:
                    desc = patt + desc[desc.find(i) :]
                    break

        elif "is being considered for deletion" in desc:
            desc = "No description."

        else:
            if strict:
                return ""
            # with open("test.txt", "a") as f:
            # f.write(name + " @ " + desc + "\n")

        return desc.strip()

    def text_to_instance(
        self,
        ID,
        input,
        output=None,
        **kwargs,
    ) -> Instance:  # type: ignore
        """
        注意，train的output是匹配的NL，epoch pred 中的 output 是question
        """
        fields_dict = {}
        meta_fields = {}

        input = _clean_white_spaces(input).strip()
        tokenized_source = self.tokenizer.tokenize(input)
        fields_dict["source_tokens"] = TextField(tokenized_source)
        meta_fields["source_tokens"] = tokenized_source

        meta_fields["ID"] = ID
        meta_fields["input"] = input

        # reward
        if "reward" in kwargs:
            fields_dict["reward"] = TensorField(torch.tensor(kwargs["reward"]))

        # other
        for n in [
            "ents",
            "cvt_map",
            "e0",
            "e1",
            "pred",
            "beam_logit",
            "compositionality_type",
            "jaccard_score",
            "patt_maps",
            "map",
            "level",
        ]:
            _t = kwargs.get(n, None)
            if _t:
                if isinstance(_t, dict):
                    meta_fields[n] = list(_t.items())
                else:
                    meta_fields[n] = _t

        # CHN: wqcwq-unseen
        if self.mode == "test" and hasattr(self, "test_iid_id") and self.test_iid_id:
            _id = ID.split("@")[0]
            if _id in self.test_iid_id:
                meta_fields["level"] = "iid"
            elif _id in self.test_unseen_id:
                meta_fields["level"] = "unseen"
            else:
                raise

        if output is not None:
            # output = output.lower()
            output = _clean_white_spaces(output).strip()
            tokenized_target = self.tokenizer.tokenize(output)
            fields_dict["target_tokens"] = TextField(tokenized_target)
            meta_fields["output"] = output

        fields_dict["metadata"] = MetadataField(meta_fields)

        return Instance(fields_dict)

    def apply_token_indexers(self, instance: Instance) -> None:
        # type: ignore
        instance.fields["source_tokens"]._token_indexers = self._token_indexers
        if "target_tokens" in instance.fields:
            # type: ignore
            instance.fields["target_tokens"]._token_indexers = self._token_indexers


class PathQuestionsReader(DSDatasetReader):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

    def _parse_item(self, item):
        """
        desc与ent之间是否需要特殊字符？
        """
        # dataset 推理
        if self.setting[:3] == "ds-" and not self.mode == "train":
            if self.setting == "ds-merge-desc-realent":
                # 无占位符
                # special处理！预测时候不能 ?c  替换为 the entity C is Harry
                for idx, spo in enumerate(item["spos"]):
                    e0, e0name, e0desc, p, e1, e1name, e1desc = spo
                    _id = item["ID"] + f"@pred@{idx}"

                    e0name = e0name.replace("_", " ").replace("-", " - ").replace(" Of ", " of ").title()
                    e1name = e1name.replace("_", " ").replace("-", " - ").replace(" Of ", " of ").title()

                    _input = f"<H> {e0name} <D> {e0desc} <P> {p} <T> {e1name} <D> {e1desc}"
                    # if e0name[0] == "?" or e1name[0] == "?":
                    #     continue
                    tmp = {"ID": _id, "input": _input.strip()}
                    if tmp["ID"] in self.skips:
                        yield None
                    else:
                        instance = self.text_to_instance(**tmp)
                        yield instance

            # 有占位符 使用type占位符 only-pred
            elif self.setting == "ds-merge-desc-pattent":
                for idx, _spo in enumerate(item["spos"]):
                    e0id, e0name, e0desc, p, e1id, e1name, e1desc = _spo

                    _id = item["ID"] + f"@pred@{idx}"
                    # if self.rerun_ids and _id not in self.rerun_ids:
                    #     return None

                    h_type, t_type = p.split(".")[-2:]
                    h_type, t_type = f"[{h_type}]", f"[{t_type}]"

                    e0desc = DSDatasetReader._replace_ent_in_desc(e0name, h_type, e0desc)
                    e1desc = DSDatasetReader._replace_ent_in_desc(e1name, t_type, e1desc)

                    _input = f"<H> {h_type} <D> {e0desc} <P> {p} <T> {t_type} <D> {e1desc}"
                    patt_maps = {h_type: e0name, t_type: e1name}

                    # _input = (
                    #     f"<H> [entity 1] <D> {e0desc} <P> {p} <T> [entity 2] <D> {e1desc}"
                    # )
                    # patt_maps = {"[entity 1]": e0name, "[entity 2]": e1name}

                    if e0name[0] == "?" or e1name[0] == "?":
                        continue

                    tmp = {
                        "ID": _id,
                        "input": _input.strip(),
                        "patt_maps": patt_maps,
                    }
                    instance = self.text_to_instance(**tmp)
                    yield instance

            else:
                # print(item)
                raise ValueError(f"err setting: {self.setting}")

        # 不含beams的
        elif self.setting in ["sparql-nomid", "human-NL-nodesc", "sparql-middesc"]:
            if self.ids is not None and item["ID"] not in self.ids:
                yield None
            else:
                tmp = {}
                tmp["ID"] = item["ID"]
                tmp["compositionality_type"] = "pathq"
                tmp["output"] = item["question"].lower().replace("_", " ")

                _input = ""
                if "sparql" in self.setting:
                    _input += item["sparql_nomid"]

                if "middesc" in self.setting:
                    _input += ". description: "
                    _input += " ".join(item["name_desc_map"].values())

                if "human" in self.setting:
                    # 注意！这里有 question:
                    _patt = item["sparql_NL-nodesc"][-1]
                    _inputs = item["sparql_NL-nodesc"][:-1] + item["patts_number"] + [_patt]
                    _input = " ".join(_inputs)

                tmp["input"] = _input.lower().replace("_", " ")
                if "level" in item:
                    tmp["level"] = item["level"]
                yield self.text_to_instance(**tmp)

        # final gen question
        elif self.setting in [
            "only-autoprompt",
            "sparql-autoprompt",
        ]:
            if self.ids is not None and item["ID"] not in self.ids:
                yield None
            else:
                if self.beam_sample == "top1":
                    _beams = item["final_beams"][:1]
                elif self.beam_sample == "random":
                    random.seed(self.random_seed)
                    _beam = random.choice(item["final_beams"])
                    _beams = [_beam]
                elif self.beam_sample == "jaccard":
                    simis = self.jaccard_map[item["ID"]]
                    simis = sorted(
                        [(idx, s) for idx, s in enumerate(simis)],
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    _i, jaccard_score = simis[0]
                    _beams = [item["final_beams"][_i]]
                    item["jaccard_score"] = jaccard_score
                elif self.beam_sample == "all":
                    # 逐个遍历  beam
                    if not self.top_beam:
                        print(colorful(f"USE beams top {self.top_beam} !!", color="red"))
                    _beams = item["final_beams"][: self.top_beam]
                else:
                    raise ValueError(f"err beam_sample: {self.beam_sample}")

                for i, _beam in enumerate(_beams):
                    logit, _auto_pmt = _beam
                    if not _auto_pmt:
                        continue

                    # if "sparql" not in self.setting:
                    # patts_number = item.get("patts_number", [])
                    # _auto_pmt = " ".join(beam + patts_number)
                    # else:
                    #     _auto_pmt = " ".join(beam)

                    _auto_pmt = _auto_pmt if _auto_pmt else "."
                    _auto_pmt = replaces(_auto_pmt, VAR_MAP)
                    _s = _auto_pmt.split(" ")
                    if len(_s) > 200:
                        x = 1

                    _auto_pmt = " ".join(_s[:200])

                    _input = ""

                    if "sparql" in self.setting:
                        _input += replaces(item["sparql_nomid"], SPARQL_CLEAN_MAP)

                    # if "middesc" in self.setting:
                    #     if not _input.strip().endswith("."):
                    #         _input += "."
                    #     _input += " description: "
                    #     _desc = " <end> ".join([i[1] for i in item["mids-desc"]])
                    #     _input += _desc

                    if not _input.strip().endswith("."):
                        _input += "."
                    _input += " description: "
                    _input += _auto_pmt

                    if self.decoder_instructions:
                        _output = "question: " + item["question"]
                        # _output = item["question"]  # try-3.0
                    else:
                        if not _input.strip().endswith("."):
                            _input += "."
                        _input += " question:"
                        _output = item["question"]

                    tmp = {}
                    tmp["ID"] = item["ID"] + f"@beam{i}"
                    tmp["input"] = _input.lower().replace("_", " ")
                    tmp["output"] = _output.lower().replace("_", " ")
                    tmp["beam_logit"] = logit
                    tmp["compositionality_type"] = item["compositionality_type"]
                    if "jaccard_score" in item:
                        tmp["jaccard_score"] = jaccard_score
                    if self.reward_map:
                        tmp["reward"] = (
                            float(self._reward_map[item["ID"]][i]) if item["ID"] in self._reward_map else 1.0
                        )

                    yield self.text_to_instance(**tmp)

        else:
            raise ValueError(f"err setting: {self.setting}")
