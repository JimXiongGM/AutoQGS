import os
import re

from common.common_algorithms import order_by_jaccard, topK_lists
from common.common_dataclean import STOP_WORDs
from common.common_utils import (
    jsonl_generator,
    read_json,
    read_jsonl,
    replaces,
    save_to_json,
    save_to_jsonl,
    try_pop_keys,
)
from data_helper.tokenizers import init_nltk, init_spacy

"""
update
run cvt-pred-merge prediction
这里合并、筛选beam search结果，获取pmt learning的输入
"""

NUMBERs = {"?sk7", "?sk4", "?sk3", "?sk2", "?sk0", "?sk8", "?sk1", "?sk5", "?num", "?x"}

REL_ENT_VER = 19
PENALTY = 1.0

COMBINE_PATH = "datasets/for_pmt/merge-beam/wqcwq"
os.makedirs(COMBINE_PATH, exist_ok=True)


def check_patt_beams(rerun_file=None):
    """
    find out beams that dont match input patterns, than re-run beam search with beam size 100(last step).
    usage:
        1. run check_patt_beams()  output: errids-ver-10-penalty-1.0.json
        2. rerun  python main/run_ds_bart.py -c config/DS-desc/ds-merge-desc-pattent.yml with `rerun_ids`
        3. run check_patt_beams(rerun_file="xxx-err-rerun.jsonl")
    """
    reruns = {}
    if rerun_file:
        reruns = read_jsonl(rerun_file)
        reruns = {i["ID"]: i for i in reruns}

    err_ids = []

    def _update():
        return {"err": len(err_ids)}

    # patt-ent
    for item in jsonl_generator(
        f"save/ds-bart-wiki18/ds-merge-desc-pattent-2.0/predictions-wqcwq-ver-{REL_ENT_VER}-penalty-{PENALTY}.jsonl",
        update_func=_update,
    ):
        if item["ID"] in reruns:
            item = reruns[item["ID"]]

        FLAG_good = False

        # all patts in output, OK.
        _map = {i[0]: i[1] for i in item["patt_maps"]}  # k:[e1,...]

        for beam in item["predictions"]:
            if not FLAG_good and all([True if patt in beam else False for patt in _map.keys()]):
                FLAG_good = True
                break

        if not FLAG_good:
            err_ids.append(item["ID"])

    print(f"len err_ids: {len(err_ids)}")
    if rerun_file is None:
        save_to_json(
            err_ids,
            f"save/ds-bart-wiki18/ds-merge-desc-pattent-2.0/errids-wqcwq-ver-{REL_ENT_VER}-penalty-{PENALTY}.json",
        )
    else:
        save_to_json(
            err_ids,
            f"save/ds-bart-wiki18/ds-merge-desc-pattent-2.0/errids-2-wqcwq-ver-{REL_ENT_VER}-penalty-{PENALTY}.json",
        )


topK = 5
_STOP_WORDs = set(STOP_WORDs + list(NUMBERs) + [").", "?x", "?y", "?c", "?k"])

tok = init_spacy(lower=False, keep_sent=True)


def _beam_filter(beam: str, min_len=2) -> list:
    beam = beam.lower().split(" ")
    beam = [i for i in beam if i and i not in _STOP_WORDs]
    return len(beam) >= min_len


# ------------------------------ real-ent ------------------------------- #

_mm = {i: "" for i in STOP_WORDs}


def compact_str(text):
    """
    压缩字符
    """
    text = (
        text.lower()
        .replace(" ", "")
        .replace("(the?x)", "")
        .replace("(the?c)", "")
        .replace("(the?k)", "")
        .replace("(the?y)", "")
        .replace("(the?num)", "")
    )
    text = replaces(text.lower(), _mm)
    return text


beam_datas_p1 = read_jsonl(
    "save/ds-bart-wiki18/ds-merge-desc-realent/predictions-wqcwq-ver-9-penalty-1.0.jsonl",
    max_instances=None,
)
beam_datas_p2 = read_jsonl(
    "save/ds-bart-wiki18/ds-merge-desc-realent/predictions-wqcwq-ver-19-penalty-1.0.jsonl",
    max_instances=None,
)
for i, j in zip(beam_datas_p1, beam_datas_p2):
    i["predictions"].extend(j["predictions"])
    i["beam_logits"].extend(j["beam_logits"])
    assert i["ID"] == j["ID"]
del beam_datas_p2

beam_datas = {i["ID"]: i for i in beam_datas_p1}
_total = set([i.split("@")[0] for i in beam_datas.keys()])

# 5073/70141 = 0.0723
err = 0


def _real(item):
    global err
    _map_special = {"<E> ": "", " </E>": "", " </E>n": "", "( )": "", "  ": " "}

    # cvt
    # re-map: [patt]: [ent1, ent2]. make sure: 1. patt IN; 2. all ents In
    cvt_beams = {}
    for k in item["cvt_spos"].keys():
        _id = item["ID"] + "@cvt@" + k

        # debug
        # if _id not in beam_datas:
        #     continue
        tmp = beam_datas[_id]
        entities = re.findall("<T> (.*?) <D>", tmp["input"]) + re.findall("<H> (.*?) <D>", tmp["input"])
        entities = [i.split("-")[0].strip('"') for i in entities]
        min_len = len(" ".join(entities).split(" ")) + 2

        logits_beams = list(zip(tmp["beam_logits"], tmp["predictions"]))
        logits_beams.sort(key=lambda x: len(x[1]))
        logits_beams = [list(i) for i in logits_beams][:10]

        valid_idx = []
        for idx, (logit, beam) in enumerate(logits_beams):
            if "<P>" in beam:
                continue
            beam = replaces(beam, _map_special)
            if _beam_filter(beam, min_len=min_len):
                if all([True if e.lower() in beam.lower() else False for e in entities]):
                    valid_idx.append(idx)

        if valid_idx:
            logits_beams = [logits_beams[i] for i in valid_idx]
            scores = order_by_jaccard(
                query=logits_beams[0][1].split(),
                candicates=[c[1].split() for c in logits_beams],
                reverse=False,
            )
            logits_beams = [logits_beams[0]] + [logits_beams[scores[0][0]]]
        else:
            logits_beams = []

        # replace "Harry" to "Harry (the ?c)"
        _n = sorted(item["mid_info"].items(), key=lambda x: len(x[1]["name"]), reverse=True)
        _map = {" " + v["name"] + " ": f" {v['name']} (the {k}) " for k, v in _n if k[:2] not in ["m.", "g."]}
        for i in range(len(logits_beams)):
            logits_beams[i][1] = replaces(" " + logits_beams[i][1] + " ", _map_special).strip()
            logits_beams[i][1] = replaces(" " + logits_beams[i][1] + " ", _map).strip()

        if not logits_beams:
            err += 1
        cvt_beams[k] = logits_beams

    # only-pred
    pred_beams = []
    for idx, spo in enumerate(item["pred_spos"]):
        _id = item["ID"] + f"@pred@{idx}"

        # debug
        if _id not in beam_datas:
            continue

        tmp = beam_datas[_id]
        entities = re.findall("<T> (.*?) <D>", tmp["input"]) + re.findall("<H> (.*?) <D>", tmp["input"])
        entities = [i.split("-")[0].strip('"') for i in entities]
        min_len = len(" ".join(entities).split(" ")) + 2

        logits_beams = list(zip(tmp["beam_logits"], tmp["predictions"]))
        logits_beams.sort(key=lambda x: len(x[1]))
        logits_beams = [list(i) for i in logits_beams][:10]

        valid_idx = []
        # _beam_tokens = set()
        for idx, (logit, beam) in enumerate(logits_beams):
            if "<P>" in beam:
                continue
            beam = replaces(beam, _map_special)
            if _beam_filter(beam, min_len=min_len) and all(
                [True if compact_str(e) in compact_str(beam) else False for e in entities]
            ):
                valid_idx.append(idx)
                # _b = set(beam.lower().split(" "))
                # ja = Jaccard_coefficient(_beam_tokens,_b)
                # if ja < 0.6:
                #     _beam_tokens +=_b

        if valid_idx:
            logits_beams = [logits_beams[i] for i in valid_idx]
            scores = order_by_jaccard(
                query=logits_beams[0][1].split(),
                candicates=[c[1].split() for c in logits_beams],
                reverse=False,
            )
            logits_beams = [logits_beams[0]] + [logits_beams[scores[0][0]]]
        else:
            logits_beams = []

        # replace "Harry" to "Harry (the ?c)"
        _map = {
            " " + v["name"] + " ": f" {v['name']} (the {k}) "
            for k, v in item["mid_info"].items()
            if k[:2] not in ["m.", "g."]
        }
        for i in range(len(logits_beams)):
            logits_beams[i][1] = replaces(" " + logits_beams[i][1] + " ", _map_special).strip()
            logits_beams[i][1] = replaces(" " + logits_beams[i][1] + " ", _map).strip()

        if not logits_beams:
            err += 1

        pred_beams.append(logits_beams)

    item["cvt_beams"] = cvt_beams
    item["pred_beams"] = pred_beams

    # do top K sort
    lists = [v for k, v in cvt_beams.items()] + pred_beams
    logits_list = [[abs(j[0]) for j in i] for i in lists]

    topk_idx = topK_lists(*logits_list, reverse=False, topK=topK)

    final_beams = [
        (
            i0[0],
            [lists[i1][idx][1] for i1, idx in enumerate(i0[1])],
        )
        for i0 in topk_idx
    ]
    final_beams = [list(i) for i in final_beams if i]

    final_beams = final_beams if final_beams else [[1, []]]

    # add mid desc if not occur
    _name_desc_map = dict(
        sorted(
            [
                (
                    " " + v["name"] + " ",
                    (
                        f" {v['name']} (the {k}) description: " + v["desc"] + " "
                        if k[:2] not in ["m.", "g."]
                        else " " + v["name"] + " description: " + v["desc"] + " "
                    ),
                )
                for k, v in item["mid_info"].items()
            ],
            key=lambda x: len(x[0]),
            reverse=True,
        )
    )
    for i in range(len(final_beams)):
        beam = " ".join(final_beams[i][1])
        for e, desc in _name_desc_map.items():
            if compact_str(e) not in compact_str(beam):
                beam += " " + desc
        final_beams[i][1] = beam.strip()

    assert final_beams
    item["final_beams"] = final_beams

    # drop
    # item = try_pop_keys(item, keys=['cvt_beams', 'pred_beams'])
    return item


def combine_real():
    """
    combine beams to data .
    """
    data = []

    def _update():
        return {"data": len(data), "total": len(_total)}

    def _gen():
        for index, item in enumerate(
            jsonl_generator(
                "datasets/wqcwq-all-infos-v1.0/all-cvt-pred.jsonl",
                topn=None,
                update_func=_update,
            )
        ):
            # debug
            # if item["ID"] != 'WebQTest-1001_3491a61f21d793901740c61b42467ecf':
            #     continue
            # if index % 100 != 0: continue
            yield item

    for i in _gen():
        r = _real(i)
        if r:
            data.append(r)

    print(err)

    # import functools
    # from multiprocessing import Pool, cpu_count

    # pool = Pool(cpu_count())
    # mapper = functools.partial(_real)

    # for r in pool.imap(mapper, _gen()):
    #     if r:
    #         data.append(r)

    # "datasets/for_pmt/merge-beam/wqcwq/all-realent-ver-19-penalty-1.0.jsonl"
    save_to_jsonl(data, f"{COMBINE_PATH}/all-realent-ver-{REL_ENT_VER}-penalty-{PENALTY}.jsonl")


# ---------------------------------------------------------------------- #
tokenizer = init_spacy(lower=True)

from metrics import cal_BLEU

bleu = cal_BLEU()


def _reward_with_target(item):
    tmp = {}
    tmp["ID"] = item["ID"]
    golden = tokenizer(item["question"], lemma=True)
    bleu_4 = []
    for s, _beams in item["final_beams"]:
        beam = " ".join(_beams)
        beam = tokenizer(beam, lemma=True)
        bleu4 = bleu([golden], beam, ngram=4)
        bleu_4.append(bleu4)

    tmp["bleu_4"] = bleu_4
    return tmp


def rescore_beams():
    """
    基于某一种相似度给beams打分
    beam_scores_input: 看到input，train dev test 都能预测
    beam_scores_golden: dev 和 test 不能预测
    """
    _gen = jsonl_generator(
        f"{COMBINE_PATH}/all-realent-ver-{REL_ENT_VER}-penalty-{PENALTY}.jsonl",
        topn=None,
    )

    # for i in tqdm(_gen,ncols=100):
    #     y = _reward_with_target(i)

    import functools
    from multiprocessing import Pool, cpu_count

    pool = Pool(cpu_count() - 2)
    mapper = functools.partial(_reward_with_target)

    res = []
    for r in pool.imap(mapper, _gen):
        res.append(r)

    save_path = f"{COMBINE_PATH}/all-realent-beam_scores-ver-{REL_ENT_VER}-penalty-{PENALTY}.jsonl"
    save_to_jsonl(res, save_path)


if __name__ == "__main__":
    """
    export PYTHONPATH=`pwd` && echo $PYTHONPATH
    python preprocess_pmt/wqcwq_combine_beams_real.py
    """
    # check_patt_beams()
    # check_patt_beams(
    #     rerun_file=f"save/ds-bart-wiki18/ds-merge-desc-pattent-2.0/predictions-ver-{REL_ENT_VER}-penalty-{PENALTY}-err-rerun.jsonl"
    # )
    # ----------

    # combine_real()
    rescore_beams()
