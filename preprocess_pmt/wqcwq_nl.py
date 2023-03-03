from tqdm import tqdm

from common.common_utils import read_json, save_to_json, try_pop_keys

"""
1. read datas from bart preprocess and parse spos. 即 datasets/wqcwq_KGs_v3.0.tar.gz
2. convert sparql to NL.
"""
# 引入 /domain/subtype/objtype
PATT_pred = "the {p} of {s} is {o}. "
PATT_pred_typed = "the {objtype} of {subtype} {s} is {o}. "

PATT_ans = "the {p} of {s} is the answer. "
PATT_ans_typed = "the {objtype} of {s}, which is a {subtype}, is the answer. "

PATT_orderby = "order all {cvt} by {p} in {order} and choose the first one. "
PATT_range = "the {p} of {sub} should between {t1} and {t2}. "
PATT_lt = "the {p} of {sub} is less than {t1}. "
PATT_gt = "the {p} of {sub} is greater than {t1}. "
PATT_prompt = "my question is, "
time_pred = set()


def PATT_mid(mids):
    _patt = " and ".join(mids) + " are known topic entities. "
    return _patt


def _clean_pred(pred):
    pred = pred.replace("_", " ").replace(".", " ").replace("ns:", "")
    return pred


def _get_name(mid_name_map, var_name_map):
    """
    get var and mid name
    """

    def _g(x):
        if x == "?x":
            return "the answer"
        # ?
        elif x.startswith("?"):
            return var_name_map.get(x, f"entity {x[1:]}")
        # m. g.
        else:
            return mid_name_map.get(x, "the event")

    return _g


def _find_var(bys):
    return [i for i in bys if i[0] == "?"][0]


def _parse(item, desc=False):
    # time cons
    # {(?y,gov_pos):{"from":(>,1999), "to":(<,2000)}, ...}
    event_num_cons = {}
    visited_vars = set()
    spo_infos = item["spo_infos"]
    for c in spo_infos["constrains"]:
        assert len(c) == 8
        _var = c[3]
        _op = c[5]
        _val = c[6].strip("^^xsd:dateTime").strip('"')
        _spo = [i for i in spo_infos["spos"] if i[2] == _var]
        assert len(_spo) == 1
        _spo = _spo[0]

        event_tuple = (_spo[0], _clean_pred(_spo[1]))
        event_num_cons.setdefault(event_tuple, {})
        if _op in [">", ">="]:
            event_num_cons[event_tuple].setdefault("gt", _val)
        elif _op in ["<", "<="]:
            event_num_cons[event_tuple].setdefault("lt", _val)
        else:
            pass

        visited_vars.add(_var)

    # 'order_bys': [['DESC', '(', 'xsd:datetime', '(', '?sk8', ')', ')']],
    # order by desc ( ?num ) limit 1
    orderby_cons = {}
    for c in spo_infos["order_bys"]:
        assert len(c) in [3, 4, 6, 7]
        _var = _find_var(c)  # ?sk8
        _op = "descending" if c[0] == "DESC" else "ascending"

        _spo = [i for i in spo_infos["spos"] if i[2] == _var]
        assert len(_spo) == 1
        _spo = _spo[0]
        _pred_k = _spo[1]
        event_tuple = (_spo[0], _pred_k)
        orderby_cons[event_tuple] = _op
        visited_vars.add(_var)

    # remove ``
    mid_name_map = {"ns:" + i[0]: i[2] for i in item["mids_names_spo"] if "type.object.name" in i[1]}
    var_name_map = {i[0]: i[2] for i in item["vars_names_spo"] if "type.object.name" in i[1]}
    getname = _get_name(mid_name_map, var_name_map)

    # apply pattern
    # add: mids are known ents
    # patts = [PATT_mid(mid_name_map.values())]
    patts = []
    patts_number = []

    for spo in [i for i in spo_infos["spos"] if i[2] != "?x"]:
        if spo[2] in visited_vars:
            continue
        s = getname(spo[0])
        p = spo[1].replace("ns:", "")
        o = getname(spo[2])

        p_s = p.split(".")
        if len(p_s) == 3:
            _domain, subtype, objtype = p_s
            _patt = PATT_pred_typed.format(
                objtype=_clean_pred(objtype),
                subtype=_clean_pred(subtype),
                s=s,
                o=o,
            )
        else:
            _patt = PATT_pred.format(p=p, s=s, o=o)
        patts.append(_patt)

    for e_k, e_v in event_num_cons.items():
        assert len(e_v) in [1, 2]
        e_var, e_pred = e_k
        e_name = getname(e_var)
        if len(e_v) == 2:
            _t1 = e_v["gt"]
            _t2 = e_v["lt"]
            _patt = PATT_range.format(p=e_pred, sub=e_name, t1=_t1, t2=_t2)
        elif "gt" in e_v:
            _t1 = e_v["gt"]
            _patt = PATT_gt.format(p=e_pred, sub=e_name, t1=_t1)
        else:
            _t1 = e_v["lt"]
            _patt = PATT_lt.format(p=e_pred, sub=e_name, t1=_t1)

        patts.append(_patt)
        patts_number.append(_patt)

    for t, order in orderby_cons.items():
        cvt, p = t
        cvt = getname(cvt)
        p = _clean_pred(p)
        _patt = PATT_orderby.format(cvt=cvt, p=p, order=order)
        patts.append(_patt)
        patts_number.append(_patt)

    # tgt ans
    tgt_spos = [i for i in spo_infos["spos"] if i[2] == "?x"]
    if len(tgt_spos) == 0:
        raise ValueError("len(tgt_spos) == 0")
    for spo in tgt_spos:
        p = spo[1].replace("ns:", "")
        s = getname(spo[0])

        p_s = p.split(".")
        if len(p_s) == 3:
            _domain, subtype, objtype = p_s
            _patt = PATT_ans_typed.format(
                objtype=_clean_pred(objtype),
                s=s,
                subtype=_clean_pred(subtype),
            )
        else:
            _patt = PATT_ans.format(p=p, s=s)
        patts.append(_patt)

    patts.append(PATT_prompt)
    return patts, patts_number


def sparql_to_NL(save_dir):
    for name in ["train", "dev", "test"]:
        datas = read_json(f"datasets/wqcwq_KGs_v4.0/{name}.json")

        # debug
        # datas = [i for i in datas if i["ID"] == "WebQTrn-1312.P0"]

        err = 0
        for item in tqdm(datas, desc=f"{name} ing", ncols=100, colour="green"):
            item["sparql_NL-nodesc"], item["patts_number"] = _parse(item, desc=False)
            item["sparql_NL-desc"], _ = _parse(item, desc=True)
            item = try_pop_keys(
                item,
                keys=[
                    "mids_types_spo",
                    "mids_descs_spo",
                    # "mids_names_spo",
                    "vars_types_spo",
                    "vars_descs_spo",
                    # "vars_names_spo",
                ],
            )

        print(f"err {name}: {err}")
        save_to_json(datas, f"{save_dir}/{name}.json")


if __name__ == "__main__":
    """
    export PYTHONPATH=`pwd` && echo $PYTHONPATH
    python preprocess_pmt/wqcwq_nl.py
    先run preprocess_bart/wqcwq.py 再run这里
    2.0: 增加mid的desc，拆分pred的 domain subtype objtype
    2.2 增加 patts_number
    2.3 no desc和desc放一起
    3.0 variable 替换为 entity ；增加mid-desc
    """
    sparql_to_NL(save_dir="datasets/for_bart/wqcwq_NLs_v4.0")
