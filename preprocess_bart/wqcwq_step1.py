import re
from collections import defaultdict

from tqdm import tqdm

from common.common_utils import read_json, save_to_json, save_to_jsonl, split_train_dev_test, try_pop_keys
from data_helper import FreebaseWrapper, SPARQLGraph
from utils.helper import replace_mid_in_sparql

"""
1. convert WebQSP format to CWQ.
2. replace mid to name in sparql
3. add KG indfos to sparql. out: `wqcwq_KGs_2.0`

Output:
    - datasets/wqcwq-all-infos-v1.0/all-cvt-pred.jsonl
"""


def data_combine(name="train"):
    if name == "train":
        wq_datas = read_json("datasets/wqcwq/WebQSP.train.json")["Questions"]
        cwq_datas = read_json("datasets/wqcwq/ComplexWebQuestions_train.json")
    else:
        wq_datas = read_json("datasets/wqcwq/WebQSP.test.json")["Questions"]
        cwq_datas = read_json("datasets/wqcwq/ComplexWebQuestions_dev.json")

    new_wq_datas = []
    for item in wq_datas:
        tmp = {}
        tmp["question"] = item["RawQuestion"]
        tmp["webqsp_id"] = item["QuestionId"]

        # Parse items
        item = item["Parses"][0]

        tmp["sparql"] = item["Sparql"]
        tmp["ID"] = item["ParseId"]

        ans = [{"answer_id": i["AnswerArgument"], "answer": i["EntityName"]} for i in item["Answers"]]
        tmp["answers"] = ans

        tmp["compositionality_type"] = "simple"
        new_wq_datas.append(tmp)

    new_cwq_datas = []
    for item in cwq_datas:
        tmp = {}
        tmp["question"] = item["question"]
        tmp["ID"] = item["ID"]
        tmp["webqsp_id"] = item["webqsp_ID"]
        tmp["sparql"] = item["sparql"]

        ans = [{"answer_id": i["answer_id"], "answer": i["answer"]} for i in item["answers"]]
        tmp["answers"] = ans

        tmp["compositionality_type"] = item["compositionality_type"]
        new_cwq_datas.append(tmp)

    wqcwq_datas = sorted(new_wq_datas + new_cwq_datas, key=lambda x: x["webqsp_id"])

    folder = "preprocess_bart/wqcwq"
    if name == "train":
        print("train", len(wqcwq_datas))
        save_to_json(wqcwq_datas, f"{folder}/train.json")
    else:
        dev, test = split_train_dev_test(wqcwq_datas, ratio=[0.5, 0.5])
        save_to_json(dev, f"{folder}/dev.json")
        save_to_json(test, f"{folder}/test.json")
        print("dev", len(dev))
        print("test", len(test))


fb = FreebaseWrapper(end_point="http://localhost:8890/sparql")
fb.set_es()


def _sparql_process(item):
    """replace mid to KG name"""
    sparql_nomid = replace_mid_in_sparql(sparql=item["sparql"], fb=fb)
    item["sparql_nomid"] = sparql_nomid
    return item


def replace_mids():
    folder = "preprocess_bart/wqcwq/"
    for name in ["train.json", "dev.json", "test.json"]:
        path = folder + name
        data = read_json(path)
        for item in tqdm(data, desc=f"processing {name}", ncols=100):
            item = _sparql_process(item)
        save_to_json(data, path)


_patt_var = "([\?].*?)[ \\n\\t)}]"
_patt_select = "(SELECT DISTINCT .*? WHERE)"


def _add_var_infos(sparql):
    """find all mids and execute sparql to get the KG infos."""
    _vars = sorted(set(re.findall(_patt_var, sparql)) - {"?sk0", "?sk1", "?sk2", "?sk3"})
    sparql_vars = re.sub(
        _patt_select,
        f"SELECT DISTINCT {' '.join(_vars)} WHERE",
        sparql.replace("\n", " "),
    )
    res = fb.query(sparql_vars)
    if not res:
        print(sparql_vars)
        return None
    else:
        # random sample one
        res_one = res[0]

    vars_names_spo = []
    vars_types_spo = []
    vars_descs_spo = []

    for _var in _vars:
        var = _var[1:]
        if var not in res_one:
            continue
        var_mid = res_one[var]["value"].replace("http://rdf.freebase.com/ns/", "")

        # name
        res_names = fb.find_obj_by_id_and_pred(mid=var_mid, predicate="type-name", return_list=True)
        vars_names_spo.extend([(_var, f"{i[0]}", f"{i[1]}") for i in res_names])

        # type
        res_types = fb.find_obj_by_id_and_pred(mid=var_mid, predicate="type-type", return_list=True)
        vars_types_spo.extend([(_var, f"{i[0]}", f"{i[1]}") for i in res_types])

        # desc update: only use the desc preprocessed by es
        res_desc = fb.get_desc_spacy(var_mid)
        vars_descs_spo.append((_var, "common.topic.description", res_desc.strip()))

    var_res_one = {f"?{k}": v["value"].replace("http://rdf.freebase.com/ns/", "") for k, v in res_one.items()}
    return var_res_one, vars_names_spo, vars_types_spo, vars_descs_spo


_patt_mid = "([mg]\..*?)[ \\n\\t)]"


def _add_mid_infos(sparql):
    mids = sorted(set(re.findall(_patt_mid, sparql)))
    mids_names_spo = []
    mids_types_spo = []
    mids_descs_spo = []

    for mid in mids:
        # name
        res_names = fb.find_obj_by_id_and_pred(mid=mid, predicate="type-name", return_list=True)
        mids_names_spo.extend([(mid, f"{i[0]}", f"{i[1]}") for i in res_names])

        # type
        res_types = fb.find_obj_by_id_and_pred(mid=mid, predicate="type-type", return_list=True)
        mids_types_spo.extend([(mid, f"{i[0]}", f"{i[1]}") for i in res_types])

        # desc
        res_descs = fb.get_desc_spacy(mid=mid).strip('"')
        mids_descs_spo.extend(
            [
                (
                    mid,
                    f"common.topic.description",
                    f"{res_descs}" if res_descs else "",
                )
            ]
        )

    return mids_names_spo, mids_types_spo, mids_descs_spo


def _clean_type(pred):
    r = pred.split(".")[-1].replace("_", " ").strip("")
    return r


DATA_INFOS = "datasets/wqcwq-all-infos-v1.0/all-cvt-pred.jsonl"


def add_KG_infos(force_run=False):
    _type_clean = lambda x: x.replace("http://rdf.freebase.com/ns/", "").split(".")[-1]
    data = [
        i
        for name in ["train.json", "dev.json", "test.json"]
        for i in read_json("preprocess_bart/wqcwq/" + name)
    ]
    err_c = 0
    assert "sparql_nomid" in data[0], "run `replace_mids()` first."
    for item in tqdm(data, ncols=100):
        graph = SPARQLGraph.build_from_cwq_item(item, to_graph=False)
        if isinstance(graph, str):
            continue

        item["spo_infos"] = {}
        item["spo_infos"]["constrains"] = graph.constrains
        item["spo_infos"]["select_list"] = graph.select_list
        item["spo_infos"]["spos"] = graph.spos
        item["spo_infos"]["order_bys"] = graph.order_bys
        item["spo_infos"]["limit"] = graph.limit

        # force re-run
        if force_run:
            try_pop_keys(item, ["vars_names_spo", "mids_names_spo"])

        if "vars_names_spo" not in item:
            res = _add_var_infos(item["sparql"])
            if res is None:
                err_c += 1
                continue
            else:
                var_res_one, vars_names_spo, vars_types_spo, vars_descs_spo = res
            item["var_res_one"] = var_res_one
            item["vars_names_spo"] = vars_names_spo
            item["vars_types_spo"] = vars_types_spo
            item["vars_descs_spo"] = vars_descs_spo
        else:
            vars_names_spo = item["vars_names_spo"]
            vars_types_spo = item["vars_types_spo"]
            vars_descs_spo = item["vars_descs_spo"]

        if "mids_names_spo" not in item:
            mids_names_spo, mids_types_spo, mids_descs_spo = _add_mid_infos(item["sparql"])
            item["mids_names_spo"] = mids_names_spo
            item["mids_types_spo"] = mids_types_spo
            item["mids_descs_spo"] = mids_descs_spo
        else:
            mids_names_spo = item["mids_names_spo"]
            mids_types_spo = item["mids_types_spo"]
            mids_descs_spo = item["mids_descs_spo"]

        # add mid type and desc
        mid_name_map = {i[0]: i[2] for i in mids_names_spo if "type.object.name" in i[1]}

        _mids_types_map = defaultdict(list)
        for ll in mids_types_spo:
            if (
                "http://rdf.freebase.com/ns/base." not in ll[2]
                and "http://rdf.freebase.com/ns/common." not in ll[2]
            ):
                _mids_types_map[ll[0]].append(_type_clean(ll[2]))

        _mids_desc = []
        for ll in mids_descs_spo:
            mid, pred, desc = ll
            if desc:
                pass
            elif mid_name_map.get(mid, "") and _mids_types_map.get(mid, []):
                _t = ", ".join([_clean_type(i) for i in _mids_types_map[mid]])
                desc = f"{mid_name_map[mid]} is a kind of {_t} ."

            _mids_desc.append((mid, desc))

        # var
        vars_name_map = {i[0]: i[2] for i in vars_names_spo if "type.object.name" in i[1]}

        _vars_types_map = defaultdict(list)
        for ll in vars_types_spo:
            if (
                "http://rdf.freebase.com/ns/base." not in ll[2]
                and "http://rdf.freebase.com/ns/common." not in ll[2]
            ):
                _vars_types_map[ll[0]].append(_type_clean(ll[2]))

        _vars_desc = []
        for ll in vars_descs_spo:
            mid, pred, desc = ll
            if desc:
                pass
            elif vars_name_map.get(mid, "") and _vars_types_map.get(mid, []):
                _t = ", ".join([_clean_type(i) for i in _vars_types_map[mid]])
                desc = f"{vars_name_map[mid]} is a kind of {_t} ."

            _vars_desc.append((mid, desc))

        item["mids-desc"] = _mids_desc
        item["vars-desc"] = _vars_desc

    print(f"err:\tlen:{err_c}\tratio:{err_c/len(data):.4f}")
    save_to_jsonl(data, DATA_INFOS)


if __name__ == "__main__":
    data_combine("train")
    data_combine("dev")
    replace_mids()

    add_KG_infos(force_run=True)
