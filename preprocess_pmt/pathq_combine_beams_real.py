import os

from common.common_algorithms import topK_lists
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

pathQ 特殊处理，pathQ没有提供mid，只能通过ES+规则匹配进行实体消歧，很有可能tmp
"""

NUMBERs = {"?sk7", "?sk4", "?sk3", "?sk2", "?sk0", "?sk8", "?sk1", "?sk5", "?num"}

PATT_ENT_VER = 20
PENALTY = 1.0

COMBINE_PATH = "datasets/for_pmt/merge-beam/pathq"
os.makedirs(COMBINE_PATH, exist_ok=True)


def check_patt_beams(rerun_file=None):
    """
    find out beams that dont match input patterns, than re-run beam search with beam size 100(last step).
    usage:
        1. run check_patt_beams()  output: errids-ver-10-penalty-1.0.json
        2. rerun  python main/run_ds_bart.py -c config/DS-desc/ds-merge-desc-realent.yml with `rerun_ids`
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
        f"save/ds-bart-wiki18/ds-merge-desc-realent-2.0/predictions-pathq-ver-{PATT_ENT_VER}-penalty-{PENALTY}.jsonl",
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
            f"save/ds-bart-wiki18/ds-merge-desc-realent-2.0/errids-pathq-ver-{PATT_ENT_VER}-penalty-{PENALTY}.json",
        )
    else:
        save_to_json(
            err_ids,
            f"save/ds-bart-wiki18/ds-merge-desc-realent-2.0/errids-2-pathq-ver-{PATT_ENT_VER}-penalty-{PENALTY}.json",
        )


topK = 5
_STOP_WORDs = set(STOP_WORDs + [").", "?x", "?y", "?c", "?k"])

tok = init_spacy(lower=False, keep_sent=True)


_STOP_WORDs = set(STOP_WORDs + list(NUMBERs) + [").", "?x", "?y", "?c", "?k"])

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
    text = replaces(text, _mm)
    return text


def _beam_filter(beam: str, min_len=2) -> list:
    beam = beam.split(" ")
    beam = [i for i in beam if i and i not in _STOP_WORDs]
    return len(beam) >= min_len


beam_datas = read_jsonl(
    f"save/ds-bart-wiki18/ds-merge-desc-realent/predictions-pathq-ver-{PATT_ENT_VER}-penalty-{PENALTY}.jsonl",
    max_instances=None,
)
beam_datas = {i["ID"]: i for i in beam_datas}
_total = set([i.split("@")[0] for i in beam_datas.keys()])
err = 0

_map_special = {"<E> ": "", " </E>": "", " </E>n": "", "( )": "", "  ": " "}


def _real(item):
    FLAG_skip = False
    global err

    _map = {n.replace("_", " "): f"{n.replace('_',' ')} (the {_var})" for n, _var in item["name_var"].items()}

    # only-pred
    pred_beams = []
    for idx, _spo in enumerate(item["spos"]):
        _id = item["ID"] + f"@pred@{idx}"

        # debug
        if _id not in beam_datas:
            FLAG_skip = True
            break

        tmp = beam_datas[_id]
        logits_beams = list(zip(tmp["beam_logits"], tmp["predictions"]))
        logits_beams.sort(key=lambda x: len(x[1]))
        logits_beams = [list(i) for i in logits_beams]

        # min_len = len(_map) + 2
        # valid_idx = []
        for idx, (logit, beam) in enumerate(logits_beams):
            logits_beams[idx][1] = replaces(logits_beams[idx][1], maps=_map_special)
            # if _beam_filter(beam, min_len=min_len) and all(
            #     [True if p.lower() in beam.lower() else False for p in _map.keys()]
            # ):
            logits_beams[idx][1] = replaces(logits_beams[idx][1], maps=_map, ignorecase=True)
            # valid_idx.append(idx)

        # if valid_idx:
        #     logits_beams = [logits_beams[i] for i in valid_idx][:5]
        #     pred_beams.append(logits_beams)
        # else:
        #     err += 1

        pred_beams.append(logits_beams[:5])

    if FLAG_skip:
        return None

    item["pred_beams"] = pred_beams

    # do top K sort
    lists = pred_beams
    logits_list = [[abs(j[0]) for j in i] for i in lists]

    topk_idx = topK_lists(*logits_list, reverse=False, topK=topK)

    final_beams = [
        [
            i0[0],
            [lists[i1][idx][1] for i1, idx in enumerate(i0[1])],
        ]
        for i0 in topk_idx
    ]

    # add mid desc if not occur
    _name_desc_map = {}
    for i in item["spos"]:
        for idx in [1, 5]:
            if i[idx] in item["name_var"]:
                _name_desc_map[i[idx]] = (
                    i[idx].replace("_", " ") + f" (the {item['name_var'][i[idx]]}) : " + i[idx + 1]
                )
            else:
                _name_desc_map[i[idx]] = i[idx].replace("_", " ") + " : " + i[idx + 1]

    # _name_desc_map = dict(
    #     sorted(
    #         [(" " + _spo[1] + " ", " " + _spo[1] + " description: " + _spo[2] + " ",),
    #         (" " + _spo[5] + " ", " " + _spo[5] + " (the ?x) description: " + _spo[6] + " ",)
    #         ],
    #         key=lambda x: len(x[0]),
    #         reverse=True,
    #     )
    # )
    for i in range(len(final_beams)):
        beam = " ".join(final_beams[i][1])
        for e, desc in _name_desc_map.items():
            if compact_str(e) not in compact_str(beam):
                beam += " " + desc
        final_beams[i][1] = beam.strip()

    # assert final_beams

    item["name_desc_map"] = _name_desc_map
    item["final_beams"] = final_beams

    # drop
    item = try_pop_keys(item, keys=["pred_beams"])
    return item


def combine_real():
    """
    combine beams to datas and make sure all pattens can be replaced back.
    """

    datas = []

    def _update():
        return {"datas": len(datas), "total": len(_total)}

    def _gen():
        for item in jsonl_generator(
            "datasets/PathQuestion-all-infos-v1.0/all.jsonl",
            topn=None,
            update_func=_update,
        ):
            # debug
            # if item["ID"] != "PQ-2H@9":
            #     continue
            yield item

    for i in _gen():
        r = _real(i)
        if r:
            datas.append(r)

    global err
    print(err)

    # import functools
    # from multiprocessing import Pool, cpu_count

    # pool = Pool(cpu_count())
    # mapper = functools.partial(_real)

    # for r in pool.imap(mapper, _gen()):
    #     if r:
    #         datas.append(r)

    # print(len(errs))
    save_to_jsonl(datas, f"{COMBINE_PATH}/all-realent-ver-{PATT_ENT_VER}-penalty-{PENALTY}.jsonl")


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
        f"{COMBINE_PATH}/all-realent-ver-{PATT_ENT_VER}-penalty-{PENALTY}.jsonl",
        topn=None,
    )

    # for i in tqdm(_gen,ncols=100):
    #     y = _reward_with_target(i)

    import functools
    from multiprocessing import Pool, cpu_count

    pool = Pool(cpu_count())
    mapper = functools.partial(_reward_with_target)

    res = []
    for r in pool.imap(mapper, _gen):
        res.append(r)

    save_path = f"{COMBINE_PATH}/all-realent-beam_scores-ver-{PATT_ENT_VER}-penalty-{PENALTY}.jsonl"
    save_to_jsonl(res, save_path)


if __name__ == "__main__":
    """
    export PYTHONPATH=`pwd` && echo $PYTHONPATH
    python preprocess_pmt/pathq_combine_beams_real.py
    """
    # check_patt_beams()
    # check_patt_beams(
    #     rerun_file=f"save/ds-bart-wiki18/ds-merge-desc-realent-2.0/predictions-ver-{PATT_ENT_VER}-penalty-{PENALTY}-err-rerun.jsonl"
    # )

    combine_real()

    # rescore_beams()

    # ----------

    # debug_beam_qualify()
    # _checkk()
