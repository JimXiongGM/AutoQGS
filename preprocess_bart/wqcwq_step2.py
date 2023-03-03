import functools
import json
import random
from collections import Counter, defaultdict
from copy import deepcopy
from multiprocessing.dummy import Pool

from elasticsearch import Elasticsearch
from tqdm import tqdm

from common.common_date import time_convert
from common.common_tokenizers import to_sent
from common.common_utils import (
    SENT_SPLIT,
    jsonl_generator,
    read_json,
    replaces,
    save_to_json,
    save_to_jsonl,
    try_pop_keys,
)
from data_helper import MIDSTART, FreebaseWrapper

"""
直接找数据集中的 predicate
本文件整合了 predicate 和 cvt-match
cvt 和 pred 的处理方式统一，根据给定的pred，找schema，不同之处在于cvt-schema是一个子图，pred-schema就是自己
1. 给定pred，在FB中获取schema-mids集合（1w个）
2. 逐个mid查ES（100个）
3. 筛选并统计覆盖率

Output:
    - datasets/wqcwq/cvt_preds_freq.json
    - datasets/wqcwq/only_preds_freq.json
    - datasets/wqcwq/cvt_pred_mids.json
    - datasets/wqcwq/cvt_triple_full_MID.jsonl
"""
fb = FreebaseWrapper(end_point="http://192.168.4.194:8891/sparql")
fb.set_es(host="localhost:9200", index="fb_desc_spacy_cased")


# ------------------- cvt pred ------------------- #

# ----- 1. ----- #


def gen_cvt_preds():
    """
    生成数据集涉及的cvt谓词和普通谓词
    """
    cvt_preds_freq = []
    only_preds_freq = []

    for item in jsonl_generator("datasets/wqcwq-all-infos-v1.0/all-cvt-pred.jsonl", topn=None):
        # cvts
        for k, spos in item["cvt_spos"].items():
            cvt_preds = [i[0] for i in spos]
            cvt_preds_freq.extend(cvt_preds)

        # preds
        preds = [i[1] for i in item["pred_spos"]]
        only_preds_freq.extend(preds)

    cvt_preds_freq = Counter(cvt_preds_freq).most_common()
    only_preds_freq = Counter(only_preds_freq).most_common()

    print(len(cvt_preds_freq))
    print(len(only_preds_freq))

    save_to_json(cvt_preds_freq, "datasets/wqcwq/cvt_preds_freq.json")
    save_to_json(only_preds_freq, "datasets/wqcwq/only_preds_freq.json")


# ----- 2. ----- #
sparql_mids_incvt = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?cvt WHERE {{
    ?e1 ns:{pred} ?cvt .
}}"""

sparql_mids_outcvt = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?cvt WHERE {{
    ?cvt ns:{pred} ?e1 .
}}"""


def query_cvt_mids():
    cvt_preds = read_json("datasets/wqcwq/cvt_preds_freq.json")
    cvt_preds = [i[0] for i in cvt_preds]
    cvt_pred_mids = {}
    for pred in tqdm(cvt_preds, ncols=80):
        # in
        if pred.startswith("R@"):
            ss = sparql_mids_incvt.format(pred=pred[2:])
        # out
        else:
            ss = sparql_mids_outcvt.format(pred=pred)

        res = fb.query(ss)
        if res:
            mids = [i["cvt"]["value"].replace("http://rdf.freebase.com/ns/", "") for i in res]
            cvt_pred_mids[pred] = mids
        else:
            print()
    save_to_json(cvt_pred_mids, "datasets/wqcwq/cvt_pred_mids.json")


# ----- 3. ----- #
sparql_base1 = """PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?e1 ?p_in WHERE {{
        ?e1 ?p_in ns:{mid} .
    }}"""
sparql_base2 = """PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?p_out ?e2 WHERE {{
        ns:{mid} ?p_out ?e2 .
    }}"""


def _get_name(mid):
    res = fb.get_mid_name(mid)
    res = res if res else mid
    return res


def q_one(mid):
    try:
        # in
        ss = sparql_base1.format(mid=mid)
        res = fb.query(ss)
        if res:
            p_in = [i["p_in"]["value"].replace("http://rdf.freebase.com/ns/", "") for i in res]

            e1 = [i["e1"]["value"].replace("http://rdf.freebase.com/ns/", "") for i in res]
            e1_name = [_get_name(i) for i in e1]
            assert len(p_in) == len(e1_name)
            in_cvt = [(eid, ename, p) for eid, ename, p in zip(e1, e1_name, p_in)]
        else:
            in_cvt = []

        # out
        ss = sparql_base2.format(mid=mid)
        res = fb.query(ss)
        if res:
            p_out = [i["p_out"]["value"].replace("http://rdf.freebase.com/ns/", "") for i in res]

            e2 = [i["e2"]["value"].replace("http://rdf.freebase.com/ns/", "") for i in res]
            e2_name = [_get_name(i) for i in e2]
            assert len(p_out) == len(e2_name)
            out_cvt = [(p, eid, ename) for p, eid, ename in zip(p_out, e2, e2_name)]
        else:
            out_cvt = []

        if (not in_cvt) and (not out_cvt):
            return None
        else:
            return {
                "mid": mid,
                "in": in_cvt,
                "out": out_cvt,
            }
    except Exception as e:
        print(mid, e)
        return None


topn = None

CVT_TRIPLE_FROM_FB = f"datasets/wqcwq/cvt_triple_full_MID.jsonl"


def query_triple():
    """
    Given a CVT, retrieve all non-type nodes surrounding the specified mid.
    """
    cvt_pred_mids = read_json("datasets/wqcwq/cvt_pred_mids.json")
    all_mids = set([mid for pred, mids in cvt_pred_mids.items() for mid in mids[:topn]])
    pbar = tqdm(total=len(all_mids), ncols=80, leave=True, position=0)

    pool = Pool(10)
    mapper = functools.partial(q_one)
    succ = 0

    with open(CVT_TRIPLE_FROM_FB, "w", encoding="utf-8") as f1:
        for r in pool.imap(mapper, all_mids):
            if r:
                succ += 1
                f1.write(json.dumps(r) + "\n")
                if succ % 1000 == 0:
                    f1.flush()
                pbar.set_postfix_str(s=f"succ: {succ}.")
            pbar.update()

    print("DONE")


# ---------------------------------------- #

es = Elasticsearch(hosts=["localhost:9200"], timeout=30, max_retries=10, retry_on_timeout=True)


def _clean_time(tt):
    tt = time_convert(
        from_formats=[
            "%Y-%m-%d",
        ],
        to_format="%Y",
        in_time=tt,
    )
    return tt


SKIPPREDS = ["freebase.valuenotation.has_no_value", "freebase.valuenotation.has_value"]


def _dsl_strategy_1(item):
    """
    所有的values都要匹配; no_value 的谓词 不要
    1. 对 in 和 out 统一处理 (遇到mid 尝试查 rate)
    若pred和value是 ("type.object.type", "olympics.demonstration_event_athlete_relationship") 则保存 value 但是不在ES中查。
    """
    # 去掉 mid
    ins = [
        ("R@" + p, e1name)
        for e1, e1name, p in item["in"]
        if e1name[:2] not in MIDSTART and p not in SKIPPREDS and p not in TYPE_PREDs
    ]
    outs = [
        (p, e1name)
        for p, e1, e1name in item["out"]
        if e1name[:2] not in MIDSTART and p not in SKIPPREDS and p not in TYPE_PREDs
    ]
    cvt_preds = ins + outs

    # number
    ents = [e1name.replace(".0", "") for p, e1name in cvt_preds]

    # 去掉 type node
    # ents = [i for i in ents if _valid(i)]

    # 归一化时间
    ents = [_clean_time(i) for i in ents]

    ents = sorted(set(ents))

    # debug
    for e in ents:
        if e.lower() == "kenneth amis":
            x = 1

    _musts = [{"match_phrase": {"content": i}} for i in ents]
    if len(_musts) > 16 or len(_musts) < 2:
        return None, None

    dsl = {"size": 100, "query": {"bool": {"must": _musts}}}

    # item.pop("in")
    # item.pop("out")
    item["cvt_preds"] = cvt_preds
    item["ents"] = ents
    return item, dsl


def _clean_content_1(text):
    """
    匹配CVT不需要太严格地分句
    """
    # text = replaces(text, _map)
    text = text.replace("( )", "")
    return text


ser = to_sent(return_list=True)


def _sent_strategy_1(ents, res):
    """
    包含所有ents的最小句子集合 不一定连续
    """
    ents = [f" {i} ".lower() for i in ents]
    contents = [_clean_content_1(r["_source"]["content"]) for r in res]
    valid_contents = []
    for content in contents:
        valid_sents = []
        # check again!
        sents = content.split(SENT_SPLIT)
        for sent in sents:
            _sent = f" {sent} ".lower()
            for ent in ents:
                if ent in _sent:
                    valid_sents.append(sent)
                    break

        # valid content 确保所有ent都在
        content = " " + " ".join(valid_sents).strip() + " "
        if not any([True for e in ents if e not in content.lower()]):
            content = content.replace(SENT_SPLIT, "").strip()
            valid_contents.append(content)
    return valid_contents


ES_INDEX = "wikipedia2018_v1.1"


def _parse_one_cvt(line):
    try:
        if isinstance(line, str):
            line = json.loads(line)
        line, dsl = _dsl_strategy_1(line)
        if not dsl:
            return None
        res = es.search(index=ES_INDEX, **dsl)["hits"]["hits"]
        if res:
            contents = _sent_strategy_1(line["ents"], res)
            if contents:
                line["contents"] = contents
                return line
    except Exception as e:
        print(e)
    return None


ES_match_cvt = f"datasets/wqcwq/cvt_triple_full_ES_wiki18-cased.jsonl"


def cvt_match_es_multi():
    from multiprocessing import Pool

    # Time spent on spacy, 10k items, 30 cores, costs 8min. Must preprocess the entire wiki
    pool = Pool(16)
    mapper = functools.partial(_parse_one_cvt)

    succ = 0

    def _update():
        return {"success": succ}

    with open(ES_match_cvt, "w", encoding="utf-8") as f2:
        for line in pool.imap(mapper, jsonl_generator(CVT_TRIPLE_FROM_FB, postfix_func=_update)):
            if line:
                succ += 1
                f2.write(json.dumps(line) + "\n")
                if succ % 1000 == 0:
                    f2.flush()


# 6. ---------------------------------------- #

ENT_template = "[entity {i}]"

INVALID_ENTs = ["from", "to"]


def _do_one(item):
    _ents = [i for i in item["ents"] if i not in INVALID_ENTs]
    if len(_ents) <= 1:
        return []

    # mid-desc-map
    _mid_info = {}
    for i in item["in"]:
        mid, name, p = i
        if mid[:2] in MIDSTART:
            _mid_info.setdefault(mid, {})
            _mid_info[mid]["name"] = name
    for i in item["out"]:
        p, mid, name = i
        if mid[:2] in MIDSTART:
            _mid_info.setdefault(mid, {})
            _mid_info[mid]["name"] = name

    # get-desc
    for mid in _mid_info.keys():
        _mid_info[mid]["desc"] = fb.get_desc_spacy(mid)
    item["mid_info"] = _mid_info

    try_pop_keys(item, keys=["cvt_preds", "ents"])
    tmps = []

    # check again
    contents = [i for i in item.pop("contents") if len(i.split(" ")) < 256 and len(i) > 20]

    for c_i, content in enumerate(contents):
        _content = f" {content} ".lower()
        if any([True for e in _ents if f" {e.strip().lower()} " not in _content]):
            continue

        tmp = deepcopy(item)
        tmp["cont_id"] = c_i
        tmp["content"] = content.strip()
        tmps.append(tmp)
    return tmps


def gen_ds_cvt_datas():
    """
    CVT matches from ES need further filtering:
    1. Number of entities must be greater than 1
    Training:
        - All clusters must have been seen before
        - Replace entities in content with special characters
    """
    data = []

    def _update():
        return {"len": len(data)}

    def _gen(topn=None):
        for item in jsonl_generator(ES_match_cvt, postfix_func=_update, topn=topn):
            yield item

    # debug
    # for i in _gen(topn=100):
    #     tmp = _do_one(i)

    from multiprocessing import Pool

    pool = Pool(16)
    mapper = functools.partial(_do_one)

    for r in pool.imap(mapper, _gen()):
        data.extend(r)

    _len = lambda x: len(str(x["in"] + x["out"])) + len(x["content"])
    data.sort(key=_len, reverse=True)
    save_to_jsonl(data, "datasets/ds-wiki18/cvt-wqcwq-full-DESC-longest-v2.0.jsonl")


# 5.


def cvt_fliter_by_freq():
    """
    Dynamic selection of data and statistics
    """
    data = list(jsonl_generator("datasets/ds-wiki18/cvt-wqcwq-full-DESC-longest-v2.0.jsonl", topn=None))
    random.seed(123)
    random.shuffle(data)

    err = 0
    for i in range(len(data)):
        if len(data[i]["in"] + data[i]["out"]) > 16:
            data[i] = None
            err += 1

    data = [i for i in data if i]
    print(f"too long: {err}")

    pred_counter = defaultdict(int)
    for item in data:
        preds = set([i[-1].lower() for i in item["in"]] + [i[0].lower() for i in item["out"]])
        for p in preds:
            pred_counter[p] += 1
    ss = {}
    ss["freq"] = sorted(pred_counter.items(), key=lambda x: x[1], reverse=True)
    save_to_json(ss, "datasets/ds-wiki18/cvt-wqcwq-full-v2.0.json")

    cvt_preds_freq = dict(read_json("datasets/wqcwq/cvt_preds_freq.json"))
    cvt_preds_freq = {k.lower(): v for k, v in cvt_preds_freq.items()}
    data_total = sum(cvt_preds_freq.values())
    data_weight = {k: v / data_total for k, v in cvt_preds_freq.items()}

    ds_total = len(data)
    tgt_ds_pred_freq = {k: min(max(int(v * ds_total) + 1, 500), 5000) for k, v in data_weight.items()}

    valid_idx = set()
    for idx, item in enumerate(tqdm(data, ncols=100, desc="caling", colour="blue")):
        _p_set = set(["r@" + i[-1].lower() for i in item["in"]] + [i[0].lower() for i in item["out"]])
        if sum([tgt_ds_pred_freq.get(p, 0) for p in _p_set]) == 0:
            continue
        for p in _p_set:
            if p in tgt_ds_pred_freq:
                tgt_ds_pred_freq[p] = max(tgt_ds_pred_freq[p] - 1, 0)
        valid_idx.add(idx)

    data = [item for idx, item in enumerate(data) if idx in valid_idx]

    cvt_preds_freq = set(list(cvt_preds_freq.keys()))
    ds_preds = set()
    for item in data:
        for p in set(["r@" + i[-1].lower() for i in item["in"]] + [i[0].lower() for i in item["out"]]):
            ds_preds.add(p)
    _inter = len(ds_preds & cvt_preds_freq)
    rate = _inter / len(cvt_preds_freq)
    print(f"Coverage: {_inter}/{len(cvt_preds_freq)}\t{rate}\tdata len:{len(data)}")

    """
    cvt-pred Coverage:
        - 463/500 0.926   data len:259310
    """

    # return
    pred_counter = defaultdict(int)
    for item in data:
        preds = set([i[-1].lower() for i in item["in"]] + [i[0].lower() for i in item["out"]])
        for p in preds:
            pred_counter[p] += 1
    ss = {}
    ss["freq"] = sorted(pred_counter.items(), key=lambda x: x[1], reverse=True)
    ss["unseen"] = sorted(cvt_preds_freq - ds_preds)
    save_to_json(ss, "datasets/ds-wiki18/cvt-wqcwq-filterByWeight-v2.0.json")
    random.seed(123)
    random.shuffle(data)
    save_to_jsonl(data, "datasets/ds-wiki18/cvt-wqcwq-filterByWeight-DESC-random-v2.0.jsonl")


import re

# ------------------- pred ------------------- #
# 1.
from SPARQLWrapper import JSON, SPARQLWrapper

_clean_prefix = lambda x: x.replace("http://rdf.freebase.com/ns/", "").replace(">", "").replace("<", "")

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


def query_pred():
    """
    Given a predicate, return <s, p, o> triples
    where:
        1. s and o are both entities
        2. s is an entity, o is a value
        3. s is a value, o is an entity
    Note! There might also be:
    ('Abracadabra', kg.object_profile.prominent_type, http://rdf.freebase.com/ns/film.film)
    """
    sparql = SPARQLWrapper("http://192.168.4.194:8891/sparql", returnFormat=JSON)
    sparql.setTimeout(300)

    # preds with tail entity: start and end with <>; not start with `m.` and `g.`

    # e1, e2
    sparql_e1_e2 = """prefix ns: <http://rdf.freebase.com/ns/> 
    select distinct ?e1 ?e1_name ?e2 ?e2_name where {{
    ?e1 ns:{pred} ?e2 .
    ?e1 ns:type.object.name ?e1_name .
    ?e2 ns:type.object.name ?e2_name .
    }}
    """

    # e1, _
    sparql_e1_ = """prefix ns: <http://rdf.freebase.com/ns/> 
    select distinct ?e1 ?e1_name ?v2 where {{
    ?e1 ns:{pred} ?v2 .
    ?e1 ns:type.object.name ?e1_name .
    }}
    """

    # _, e2
    sparql__e2 = """prefix ns: <http://rdf.freebase.com/ns/> 
    select distinct ?v1 ?e2_name ?e2 where {{
    ?v1 ns:{pred} ?e2 .
    ?e2 ns:type.object.name ?e2_name .
    }}
    """
    sparqls = [sparql_e1_e2, sparql_e1_, sparql__e2]

    def _q(pred, limit=None):
        if pred in TYPE_PREDs:
            return []
        try:
            for idx, spa in enumerate(sparqls):
                sparql_txt = spa.format(pred=pred)
                if limit:
                    sparql_txt += " limit " + str(limit)
                sparql.setQuery(sparql_txt)
                res_json = sparql.query().convert()
                res_json = res_json["results"]["bindings"]
                if res_json:
                    tups = []
                    # case 1 -> (e1_name,pred,e2_name)
                    if idx == 0:
                        for rr in res_json:
                            e1 = _clean_prefix(rr["e1"]["value"])
                            e2 = _clean_prefix(rr["e2"]["value"])
                            e1_name = rr["e1_name"]["value"]
                            e2_name = rr["e2_name"]["value"]
                            tups.append((e1, e1_name, e2, e2_name))
                        return tups

                    # case 2 -> (e1_name,pred,v2)
                    elif idx == 1:
                        for rr in res_json:
                            e1 = _clean_prefix(rr["e1"]["value"])
                            e2 = ""
                            e1_name = rr["e1_name"]["value"]
                            v2 = rr["v2"]["value"]
                            tups.append((e1, e1_name, e2, v2))
                        return tups

                    # case 3 -> (v1,pred,e2_name)
                    else:
                        for rr in res_json:
                            e1 = ""
                            e2 = _clean_prefix(rr["e2"]["value"])
                            v1 = _clean_prefix(rr["e1"]["value"])
                            e2_name = rr["e2_name"]["value"]
                            tups.append((e1, v1, e2, e2_name))
                        return tups
            return []
        except Exception as e:
            print(e)
            return []

    return _q


def merge_dict(*dicts, mode="add"):
    tmp = {}
    if mode == "add":
        for _d in dicts:
            for k, v in _d.items():
                if k in tmp:
                    tmp[k] += v
                else:
                    tmp[k] = v
    elif mode == "first":
        pass
    else:
        pass

    return tmp


def merge_datasets_pred():
    fb_preds = read_json("datasets/predicate_freq.json")
    fb_preds = [[k[:-1].replace("<http://rdf.freebase.com/ns/", ""), v] for k, v in fb_preds.items()]
    fb_preds = dict(fb_preds)

    wqcwq = read_json("datasets/wqcwq/only_preds_freq.json")
    wqcwq = dict(wqcwq)
    path_ques = read_json(
        "/home/jimx/codes/KGQG_controlable/datasets/PathQuestion_v1.0/pred_freq_path_question.json"
    )
    path_ques = dict(path_ques)

    data_preds = merge_dict(wqcwq, path_ques)
    data_preds = dict(sorted(data_preds.items(), key=lambda x: x[1], reverse=True))
    for k in list(data_preds.keys()):
        if k.count(".") < 1:
            print(k)
            data_preds.pop(k)

    save_to_json(data_preds, "datasets/wqcwq_pathq_simpleq/preds_freq.json")


PRED_TRIPLE = f"datasets/wqcwq_pathq_simpleq/preds_triple_full_MID.jsonl"


def pred_match_triple_in_fb():
    """
    传入 pred list 对每一个pred查询头尾实体或者value，默认limit 1w
    topn 表示前n个谓词
    更新 增加多个数据集
    """
    only_preds_freq = read_json("datasets/wqcwq_pathq_simpleq/preds_freq.json")

    preds_dict = {}
    qer = query_pred()
    _zero = 0
    for pred in tqdm(only_preds_freq.keys(), ncols=100, colour="green"):
        res = qer(pred, limit=None)
        preds_dict[pred] = res
        if not res:
            _zero += 1
            print(f"nomid pred: {pred}")

    save_to_jsonl(preds_dict, PRED_TRIPLE)
    print("zeors:", _zero)


# 2.

_map = {"``": " ", "''": " ", ",": ".", "(": ".", ")": ".", ":": "."}


def _clean_content_2(text):
    # text = replaces(text, _map)
    return text


_p = " @e0@ (.*?) @e1@ "


def _filter_pred_nl(pred_nl):
    """
    这里判断的是 替换掉 @e0@ @e1@ 之后的pred-nl长度
    """
    if len(pred_nl) <= 2:
        return False
    # if "," in pred_nl or "(" in pred_nl or ")" in pred_nl:
    #     return False
    return True


def _match_in_sent(sent):
    """
    如果 替换掉 @e0@ @e1@ 之后的pred-nl满足条件，则返回整个句子。
    """
    sent = f" {sent} "
    pred_nls = [i.replace(" @e0@ ", "").replace(" @e1@ ", "").strip() for i in re.findall(_p, sent)]

    # filter here!
    pred_nls = [i for i in pred_nls if _filter_pred_nl(i)]

    # update!
    if pred_nls:
        return sent
    else:
        return None


def _match_es(e0, e1):
    """
    这里只缓存，不替换
    """
    # e0, e1 = e0.lower(), e1.lower()
    dsl = {
        "size": 100,
        "query": {
            "bool": {
                "must": [
                    {"match_phrase": {"content": e0}},
                    {"match_phrase": {"content": e1}},
                ]
            }
        },
    }

    pred_nls = []
    res = es.search(index=ES_INDEX, **dsl)
    for hit in res["hits"]["hits"]:
        content = _clean_content_2(hit["_source"]["content"])
        # check!
        sents = content.split(SENT_SPLIT)
        for sent in sents:
            # 之前的想法，抽取出 @e1@  @e2@ 之间的文本，如果大于两个字符则有效
            # 现在只要都在就行
            if (
                e0.lower() in sent.lower()
                and e1.lower() in sent.lower()
                and len(sent) - len(e0) - len(e1) > 4
            ):
                pred_nls.append(sent.strip())
    return pred_nls


def _parse_one_pred(tt):
    """
    input:
        - (s, p, 0)
    """
    pred, tup = tt
    e0, e0_name, e1, e1_name = tup

    # invalid
    if any([True for i in [e0_name, e1_name] if i.startswith("http://rdf.freebase.com/ns")]):
        return None

    pred_nls = _match_es(e0_name, e1_name)
    if pred_nls:
        line = (e0, e0_name, pred, e1, e1_name, pred_nls)
        return line
    else:
        return None


PRED_MATCH_ES = f"datasets/wqcwq_pathq_simpleq/preds_triples_full_ES_wiki18.jsonl"


def pred_match_es_multi():
    data = dict(list(jsonl_generator(PRED_TRIPLE, topn=None)))
    inputs = ((pred, tup) for pred, tuples in data.items() for tup in tuples)

    # debug
    # for tt in inputs:
    #     x = _parse_one_pred(tt)

    from multiprocessing import Pool

    pool = Pool(32)
    mapper = functools.partial(_parse_one_pred)

    total_v = sum([len(i) for i in data.values()])
    pbar = tqdm(total=total_v, ncols=100, desc="pred_match_es_multi")
    final = []
    for res in pool.imap(mapper, inputs):
        if res:
            final.append(res)
            pbar.set_postfix_str(f"success: {len(final)}")
        pbar.update()

    save_to_jsonl(final, PRED_MATCH_ES)


# 3.
_map1 = {"\xc2 ": "", "\xa0 ": ""}


def _clean1(text):
    return replaces(text, _map1)


DATA_LONGEST = f"datasets/ds-wiki18/wqcwq_pathq_simpleq/pred-full-DESC-longest-v1.0.jsonl"

_pred_count = defaultdict(int)


def _get_desc(item):
    _id, e0, e0_name, pred, e1, e1_name, contents = item
    tmps = []
    contents = [i for i in contents if len(i) < 600 and len(i) > 20]
    for idx, cont in enumerate(contents):
        if _pred_count[pred] > 5000:
            continue
        cont = _clean1(cont)
        if len(cont) > 600:
            x = 1
        tmp = {}
        tmp["id"] = str(_id)
        tmp["pred"] = pred
        tmp["e0"] = e0
        tmp["e1"] = e1

        tmp["e0name"] = e0_name
        tmp["e1name"] = e1_name

        tmp["e0desc"] = fb.get_desc_spacy(e0)
        tmp["e1desc"] = fb.get_desc_spacy(e1)

        tmp["cont_id"] = idx
        tmp["output"] = cont
        tmps.append(tmp)
        _pred_count[pred] += 1

    return tmps


def gen_pred_datas():
    """
    注意，这里为了加速，需要多线程共享 _pred_count 但是最后会卡住
    """
    data = []

    def _postfix_func():
        return {"success ds": len(data)}

    def _gen():
        for idx, item in enumerate(jsonl_generator(PRED_MATCH_ES, topn=None, postfix_func=_postfix_func)):
            item = [idx] + list(item)
            yield item

    # debug
    for i in _gen():
        r = _get_desc(i)
        data.extend(r)

    print(f"DONE. len: {len(data)}")

    _len = lambda x: len(x["pred"]) + len(x["output"])
    data.sort(key=_len, reverse=True)
    save_to_jsonl(data, DATA_LONGEST)


def pred_fliter_by_freq():
    """
    动态选择数据与统计
    更新 保留 wqcwq pathq simpleq
    """
    data = list(jsonl_generator(DATA_LONGEST, topn=None))
    random.seed(123)
    random.shuffle(data)

    # 过滤数据
    # 使得ds数据中pred的频率和dataset一致
    preds_freq = dict(read_json("datasets/wqcwq_pathq_simpleq/preds_freq.json"))

    # wqcwq
    # wqcwq_freq = dict(read_json("datasets/wqcwq/only_preds_freq.json"))

    preds_freq = {k.lower(): v for k, v in preds_freq.items()}

    # print(f"len of pathq and simpleq: {len(preds_freq)}")

    data_total = sum(preds_freq.values())
    data_weight = {k: v / data_total for k, v in preds_freq.items()}

    ds_total = len(data)
    tgt_ds_pred_freq = {k: min(max(int(v * ds_total) + 1, 500), 5000) for k, v in data_weight.items()}

    # 整个item中，如果所有的谓词都不需要了，则跳过
    valid_idx = set()
    for idx, item in enumerate(tqdm(data, ncols=100, desc="caling", colour="blue")):
        p = item["pred"].lower()
        if tgt_ds_pred_freq.get(p, 0) == 0:
            continue
        if p in tgt_ds_pred_freq:
            tgt_ds_pred_freq[p] = max(tgt_ds_pred_freq[p] - 1, 0)
        valid_idx.add(idx)

    data = [item for idx, item in enumerate(data) if idx in valid_idx]

    for i in data:
        i["ID"] = "pred@" + i.pop("id")
    random.shuffle(data)

    # 统计wqcwq中的pred有多少被覆盖
    preds_freq = set(list(preds_freq.keys()))
    ds_preds = set()
    for line in data:
        p = line["pred"].lower()
        ds_preds.add(p)
    _inter = len(ds_preds & preds_freq)
    rate = _inter / len(preds_freq)
    print(f"覆盖率: {_inter}/{len(preds_freq)}\t{rate}\tdata len:{len(data)}")

    """
    pred覆盖率
        - full: 1877/1983       0.9465456379223399      data len:1722481
    """

    # return
    # 统计ds谓词词频，以及unseen
    pred_counter = defaultdict(int)
    for line in data:
        p = line["pred"].lower()
        pred_counter[p] += 1
    ss = {}
    ss["freq"] = sorted(pred_counter.items(), key=lambda x: x[1], reverse=True)
    ss["unseen"] = sorted(preds_freq - ds_preds)
    save_to_json(ss, "datasets/ds-wiki18/wqcwq_pathq_simpleq/pred-full-v1.0.json")
    save_to_jsonl(
        data,
        "datasets/ds-wiki18/wqcwq_pathq_simpleq/wqcwq_pathq_simpleq-pred-filterByWeight-DESC-v1.0.jsonl",
    )


if __name__ == "__main__":
    # 1.
    gen_cvt_preds()
    query_cvt_mids()

    # 2.
    query_triple()

    # 3.
    # _debug()
    cvt_match_es_multi()

    # 4.
    # gen_ds_cvt_datas()

    # 5. filter
    cvt_fliter_by_freq()

    # ----- pred ----- #

    # 更新 pred部分增加更多的数据集
    merge_datasets_pred()

    pred_match_triple_in_fb()

    # 2. 在ES中找`(e1, e2)`对应的doc，获取p对应的NL。（更新，保存出现e1 e2的整个句子，而不是只有中间的部分）
    pred_match_es_multi()

    # 3. gen pred data
    gen_pred_datas()

    # 4. filter
    pred_fliter_by_freq()
