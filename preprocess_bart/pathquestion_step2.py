import random
from collections import Counter

from tqdm import tqdm

from common.common_utils import SENT_SPLIT, read_json, read_jsonl, save_to_json, save_to_jsonl, try_pop_keys
from data_helper.freebase_wrapper import FreebaseWrapper


def basic_convert():
    def _parse(path):
        path = path.split("#")
        if path[-2] == "<end>":
            path = path[:-2]
        assert len(path) % 2 == 1
        spos = []
        for i in range(0, len(path) - 2, 2):
            subj, pred, obj = path[i : i + 3]
            pred = pred.replace("__", ".").strip(".")
            spos.append((subj, pred, obj))
        return spos

    def spo_parse():
        fb = FreebaseWrapper(end_point="http://localhost:8890/sparql")
        fb.set_es()
        predicate_freq = read_json("datasets/predicate_freq.json")
        pred_map = {}
        for pred, freq in predicate_freq.items():
            pred = pred.replace("<http://rdf.freebase.com/ns/", "").strip(">")
            _p = pred.split(".")[-1].strip(">")
            pred_map.setdefault(_p, pred)
            pred_map.setdefault(pred, pred)

        def _do(spo):
            """
            Returns [subj_mid, original-name, subj_desc, pred, obj_mid, original-name, obj_desc]
            Note! Here, we directly use the default matching of es for entity disambiguation.
            """
            ori_subj, pred, ori_obj = spo
            subj_name, obj_name = ori_subj.replace("_", " "), ori_obj.replace("_", " ")
            subjres = fb.entity_search(subj_name)
            objres = fb.entity_search(obj_name)
            subj_mid, subj_desc, obj_mid, obj_desc = (
                None,
                "No description.",
                None,
                "No description.",
            )
            if subjres:
                _t = subjres[0]
                subj_mid = _t["_id"]
                subj_desc = _t["_source"]["desc"].split(SENT_SPLIT)[0].split("\\n")[0]
            if objres:
                _t = objres[0]
                obj_mid = _t["_id"]
                obj_desc = _t["_source"]["desc"].split(SENT_SPLIT)[0].split("\n")[0]

            _pred = pred_map[pred]
            return (
                subj_mid,
                ori_subj,
                subj_desc,
                _pred,
                obj_mid,
                ori_obj,
                obj_desc,
            )

        return _do

    def _to_sparql(spos):
        """
        Update. Replace the intermediate variables with c1 c2. Add executable sparql.
        SELECT ?x WHERE { ... }
        """
        _name_var_map = {}
        var_idx = 0
        spos = [list(i) for i in spos]
        for i in range(0, len(spos)):
            if i != 0:
                if spos[i][1] not in _name_var_map:
                    _name_var_map[spos[i][1]] = f"?c{var_idx}"
                    var_idx += 1
                spos[i][1] = _name_var_map[spos[i][1]]

            if i != len(spos) - 1:
                if spos[i][5] not in _name_var_map:
                    _name_var_map[spos[i][5]] = f"?c{var_idx}"
                    var_idx += 1
                spos[i][5] = _name_var_map[spos[i][5]]

        if spos[len(spos) - 1][5] not in _name_var_map:
            _name_var_map[spos[len(spos) - 1][5]] = f"?x"
        spos[len(spos) - 1][5] = _name_var_map[spos[len(spos) - 1][5]]

        sparql_end = "}"

        # sparql
        sparql = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?x WHERE { "
        for idx, spo in enumerate(spos):
            subj = f"ns:{spo[0]}" if spo[1][0] != "?" else spo[1]
            subj = subj if spo[0] else f"None@{spo[1]}"
            subj = subj.replace("_", " ")
            pred = "ns:" + spo[3]
            if idx == len(spos) - 1:
                obj = "?x"
            else:
                obj = f"ns:{spo[4]}" if spo[5][0] != "?" else spo[5]
                obj = obj.replace("_", " ")
            sparql += " ".join([subj, pred, obj, "."])
            sparql += " "

        sparql += sparql_end

        # no-mid
        sparql_nomid = "SELECT ?x WHERE { "
        for idx, spo in enumerate(spos):
            subj = f'"{spo[1]}"' if spo[1][0] != "?" else spo[1]
            pred = spo[3]
            if idx == len(spos) - 1:
                obj = "?x"
            else:
                obj = f'"{spo[5]}"' if spo[5][0] != "?" else spo[5]
            sparql_nomid += " ".join([subj, pred, obj, "."])
            sparql_nomid += " "

        sparql_nomid += sparql_end
        return sparql, sparql_nomid, _name_var_map

    names = ["PQ-2H", "PQ-3H", "PQL-2H", "PQL-3H", "PQL-3H_more"]
    datas = []
    spo_parser = spo_parse()
    for name in names:
        with open(f"datasets/PathQuestion/{name}.txt") as f:
            data = f.readlines()
        data = [i.strip().split("\t") for i in data]
        data = [i for i in data if i]
        for i in data:
            assert len(i) == 3
        print(name, "len: ", len(data))

        for idx, item in enumerate(tqdm(data, ncols=100, desc=name)):
            question, answer, path = item
            spos = _parse(path)
            tmp = {}
            tmp["ID"] = f"{name}@{idx}"
            tmp["question"] = question
            tmp["answer"] = answer
            tmp["compositionality_type"] = name
            tmp["path"] = path
            _spos = [spo_parser(i) for i in spos]
            tmp["spos"] = _spos
            sparql, sparql_nomid, _name_var_map = _to_sparql(_spos)
            tmp["sparql"] = sparql
            tmp["sparql_nomid"] = sparql_nomid
            tmp["name_var"] = _name_var_map

            datas.append(tmp)

    save_to_jsonl(datas, "datasets/PathQuestion-all-infos-v1.0/all.jsonl")


def split_dataset():
    """
    Toward Subgraph: 9,793/1,000/1,000
    Note! The dev and test sets are strictly aligned here. The train set has 9871 samples, which is more than expected because after deduplication, dev and test sets have fewer than 1000 samples each.
    """
    datas = read_jsonl("datasets/PathQuestion-all-infos-v1.0/all.jsonl")
    _datas_map = {i["ID"]: i for i in datas}

    test_align = read_json("preprocess_bart/PathQuestion/match_res_test.json")
    test_ids = [i["match_ID"] for i in test_align]
    tests = [_datas_map[_id] for _id in test_ids]
    print("test: ", len(tests))

    dev_align = read_json("preprocess_bart/PathQuestion/match_res_dev.json")
    dev_ids = [i["match_ID"] for i in dev_align]
    devs = [_datas_map[_id] for _id in dev_ids]
    print("dev: ", len(devs))

    ex_ids = set(test_ids + dev_ids)
    trains = [i for i in datas if i["ID"] not in ex_ids]
    print("train: ", len(trains))

    train_ids = [i["ID"] for i in trains]
    dev_ids = [i["ID"] for i in devs]
    test_ids = [i["ID"] for i in tests]

    train_ids.sort()
    random.seed(123)
    random.shuffle(train_ids)

    _id = {"train": train_ids, "dev": dev_ids, "test": test_ids}
    save_to_json(_id, "datasets/PathQuestion-all-infos-v1.0/ids_map.json")


# step 2
def pred_freq():
    datas = read_jsonl("datasets/PathQuestion-all-infos-v1.0/all.jsonl")
    preds = []
    for item in datas:
        spos = item["spos"]
        for spo in spos:
            preds.append(spo[3])
    preds = Counter(preds)
    save_to_json(
        preds.most_common(),
        "datasets/PathQuestion-all-infos-v1.0/pred_freq_path_question.json",
    )


def release_to_public():
    data = read_jsonl("datasets/PathQuestion-all-infos-v1.0/all.jsonl")
    for d in data:
        try_pop_keys(
            d,
            [
                "spos",
                "sparql_nomid",
                "compositionality_type",
            ],
        )
    ids_map = read_json("datasets/PathQuestion-all-infos-v1.0/ids_map.json")
    for name in ["train", "dev", "test"]:
        ids = set(ids_map[name])
        data_part = [d for d in data if d["ID"] in ids]
        save_to_json(data_part, f"datasets/PathQuestion_KGQG/{name}.json")


if __name__ == "__main__":
    basic_convert()
    split_dataset()

    pred_freq()

    release_to_public()
