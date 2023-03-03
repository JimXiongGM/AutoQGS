import random

from common.common_dataclean import find_special_char
from common.common_utils import (
    file_line_count,
    jsonl_generator,
    read_json,
    read_jsonl,
    save_to_json,
    save_to_jsonl,
    split_train_dev_test,
    try_pop_keys,
)
from preprocess_bart.wqcwq_step1 import DATA_INFOS
from utils.algorithms import Levenshtein_Distance

"""
1. convert WebQSP format to CWQ.
2. replace mid to name in sparql
3. add KG indfos to sparql. out: `wqcwq_KGs_2.0`
"""

# -------------------- bad ents -------------------- #


def _argmin(values):
    min_idx = -1
    min_val = 9e9
    for i, v in enumerate(values):
        min_idx = i if v < min_val else min_idx
        min_val = v if v < min_val else min_val
    return min_idx


def replace_special_token(item):
    """replace special chars in question of tokens in KG"""
    ques = item["question"]
    ents = [i[2].strip("") for i in item["mids_names_spo"]]
    ents = " ".join(ents).split(" ")
    ents = [i for i in ents if i]
    ques_list_sp = [i for i in ques.split(" ") if find_special_char(i)]
    replaces = []
    for ques_sp in ques_list_sp:
        values = [Levenshtein_Distance(ques_sp, ent) for ent in ents]
        idx = _argmin(values)
        if idx > -1:
            replaces.append((ques_sp, ents[idx]))

    item["entity_name_replaces"] = replaces

    new_q = ques
    for rep in replaces:
        new_q = new_q.replace(rep[0], rep[1])

    item["ques_replace_with_KG"] = new_q
    return item


def get_cvt_in_preds(path="datasets/cache/properties_expecting_cvt.csv", mode="http"):
    cvt_preds = set()
    with open(path) as f1:
        properties_expecting_cvt = f1.readlines()
        for pred in properties_expecting_cvt:
            pred = pred.strip().replace("/", ".")
            if mode == "http":
                pred = "/" + pred[1:]
                pred = f"<http://rdf.freebase.com/ns{pred}>"
            elif mode == "ns":
                pred = "ns:" + pred[1:]
            else:
                pred = pred[1:]
            cvt_preds.add(pred)
    return cvt_preds


DATA_INFOS_CVT_PRED = "datasets/wqcwq-all-infos-v1.0/all-cvt-pred.jsonl"

ds_cvt_freq = dict(read_json(f"datasets/ds-wiki18/cvt-wqcwq-full-v2.0.json")["freq"])
pred_cvt_freq = dict(read_json(f"datasets/ds-wiki18/wqcwq_pathq_simpleq/pred-full-v1.0.json")["freq"])
cvt_preds = get_cvt_in_preds(mode="ns")


def _parse(item):
    # get cvt spos
    if "spo_infos" not in item:
        return
    spos = item["spo_infos"]["spos"]
    cvt_vars = [spo[2] for spo in spos if spo[1] in cvt_preds]

    # break spos into cvt spos and topic spos
    # cvt spos: {"?c":(1,2)} --> spos[1] and spos[2]

    # 更新！
    names_map = {
        i[0]: i[2] for i in item["vars_names_spo"] + item["mids_names_spo"] if i[1] == "type.object.name"
    }
    desc_map = {i[0]: i[2] for i in item["vars_descs_spo"] + item["mids_descs_spo"]}
    mid_info = {}
    for mid, name in names_map.items():
        mid_info[mid] = {}
        mid_info[mid]["name"] = names_map[mid]
        mid_info[mid]["desc"] = desc_map[mid]

    item["mid_info"] = mid_info

    visited = set()
    cvt_spos = {}  # ?c:{in:[]  out:[]  constrains}

    for cvt in cvt_vars:
        tmp_ins, tmp_outs, tmp_cons = [], [], []
        for idx, spo in enumerate(spos):
            p = spo[1].replace("ns:", "")
            # out cvt
            if spo[0] == cvt:
                mid = spo[2].replace("ns:", "")
                if mid.startswith("?sk"):
                    tmp_cons.append(spo)
                else:
                    name = names_map[mid] if mid in names_map else mid.split("^^xsd:")[0]
                    tmp_outs.append([p, mid, name])
                visited.add(idx)

            # in cvt
            elif spo[2] == cvt:
                mid = spo[0].replace("ns:", "")
                if mid.startswith("?sk"):
                    tmp_cons.append(spo)
                else:
                    name = names_map[mid]
                    tmp_ins.append([mid, name, p])
                visited.add(idx)

        cvt_spos[cvt] = {"in": tmp_ins, "out": tmp_outs, "constrains": tmp_cons}

    item["cvt_spos"] = cvt_spos

    # topic ent
    _pred_spos = [spo for idx, spo in enumerate(spos) if idx not in visited]
    pred_spos = []
    for spo in _pred_spos:
        if len(spo) != 4:
            continue
        e0, p, e1, _ = spo
        e0, p, e1 = e0.replace("ns:", ""), p.replace("ns:", ""), e1.replace("ns:", "")
        pred_spos.append([e0, p, e1])

    item["pred_spos"] = pred_spos

    # add freq
    cvt_freqs = []
    pred_freqs = []
    if ds_cvt_freq and pred_cvt_freq:
        for k in cvt_spos.keys():
            _ps = [s[2] for s in cvt_spos[k]["in"]] + [s[0] for s in cvt_spos[k]["out"]]
            for p in _ps:
                p = p.lower()
                cvt_freqs.append((p, ds_cvt_freq.get(p, 0)))
        for spo in pred_spos:
            p = spo[1].lower()
            pred_freqs.append((p, pred_cvt_freq.get(p, 0)))

    # save
    item["pred_spos"] = pred_spos
    item["cvt_freqs"] = cvt_freqs
    item["pred_freqs"] = pred_freqs

    return item


from data_helper.tokenizers import init_spacy

tokenize = init_spacy(lower=False)


def parse_basic_unit():
    """
    这里输出的 wqcwq_KGs 将作为其他数据处理方法的输入。
    更新
    1. 缓存 `ques_replace_with_KG`.
    2. 识别cvt子图和普通谓词
    """
    err = 0
    new_ques = {}
    data = []
    ori_len = file_line_count(DATA_INFOS)
    for idx, item in enumerate(jsonl_generator(DATA_INFOS)):
        if "vars_names_spo" not in item:
            err += 1
            if "#MANUAL SPARQL" not in item["sparql"]:
                continue
        item = replace_special_token(item)

        item = try_pop_keys(
            item,
            keys=[
                "vars_types_spo",
                # "vars_descs_spo",
                "mids_types_spo",
                # "mids_descs_spo",
            ],
        )

        # add cvt-subgraph
        item = _parse(item)

        # spacy retoken
        tokens = tokenize(text=item["ques_replace_with_KG"])
        item["ques_spacy"] = " ".join(tokens)

        if "entity_name_replaces" in item and item["entity_name_replaces"]:
            new_ques[item["ID"]] = {}
            new_ques[item["ID"]]["new_ques"] = item["ques_replace_with_KG"]
            new_ques[item["ID"]]["entity_name_replaces"] = item["entity_name_replaces"]

        data.append(item)

    print(f"ori len: {ori_len}\tnew len: {len(data)}")
    save_to_jsonl(data, DATA_INFOS_CVT_PRED)
    save_to_json(new_ques, "datasets/cache/wqcwq_ques_replace_with_KG.json")


def resplit_v2():
    """
    Filter out unreasonable sparql
    """
    dev_test_ids = set(
        [
            i["ID"]
            for i in read_json("preprocess_bart/wqcwq/dev.json")
            + read_json("preprocess_bart/wqcwq/test.json")
        ]
    )
    data = read_jsonl(DATA_INFOS_CVT_PRED)
    valid_dev_test_ids = [i["ID"] for i in data if i["ID"] in dev_test_ids]
    valid_dev_test_ids.sort()
    dev_ids, test_ids = split_train_dev_test(valid_dev_test_ids, ratio=[0.5, 0.5])
    train_ids = list(set([i["ID"] for i in data]) - set(dev_ids) - set(test_ids))
    train_ids.sort()
    random.seed(123)
    random.shuffle(train_ids)
    _id = {"train": train_ids, "dev": list(dev_ids), "test": list(test_ids)}
    save_to_json(_id, "datasets/wqcwq-all-infos-v1.0/ids_map_v2.0.json")


def release_to_public():
    data = read_jsonl(DATA_INFOS_CVT_PRED)
    for d in data:
        try_pop_keys(
            d,
            [
                "sparql_nomid",
                "spo_infos",
                "var_res_one",
                "vars_names_spo",
                "vars_descs_spo",
                "mids_names_spo",
                "mids_descs_spo",
                "mids-desc",
                "vars-desc",
                "entity_name_replaces",
                "ques_replace_with_KG",
                "mid_info",
                "cvt_spos",
                "pred_spos",
                "cvt_freqs",
                "pred_freqs",
                "ques_spacy",
                "answers",
                "webqsp_id",
                "compositionality_type",
            ],
        )

    # unseen_wqcwq
    ids_map = read_json("idmap_unseen_wqcwq.json")
    for k, v in ids_map.items():
        ids_map[k] = set(v)

    trains, devs, tests = [], [], []
    for d in data:
        if d["ID"] in ids_map["train"]:
            trains.append(d)
        elif d["ID"] in ids_map["dev"]:
            devs.append(d)
        elif d["ID"] in ids_map["test-iid"]:
            d["compositional"] = "iid"
            tests.append(d)
        elif d["ID"] in ids_map["test-unseen"]:
            d["compositional"] = "unseen"
            tests.append(d)
        else:
            raise
    save_to_json(trains, f"datasets/WQCWQ_Unseen/train.json")
    save_to_json(devs, f"datasets/WQCWQ_Unseen/dev.json")
    save_to_json(tests, f"datasets/WQCWQ_Unseen/test.json")


if __name__ == "__main__":
    parse_basic_unit()
    resplit_v2()
    release_to_public()
