import os
from copy import deepcopy
from json import JSONDecodeError
from logging import warning
from pprint import pprint

from tqdm.std import tqdm

from common.common_utils import read_json, read_jsonl, save_to_json, try_pop_keys
from data_helper.tokenizers import init_nltk, init_spacy
from data_helper.vocabulary import Vocab


def postprocess_evaluate_batch(batch: dict):
    if isinstance(batch, list):
        return batch

    # pop no-list
    for k in list(batch.keys()):
        if not isinstance(batch[k], list):
            batch.pop(k)

    _lens = set([len(v) for k, v in batch.items()])
    assert len(_lens) == 1
    instances = []
    keys = list(batch.keys())
    _len = len(batch[keys[0]])
    for i in range(_len):
        tmp = {}
        for k in keys:
            if k == "metadata":
                for k2 in batch[k][0].keys():  # list
                    tmp[k2] = batch[k][i][k2] if k2 in batch[k][i] else 0
            else:
                tmp[k] = batch[k][i]
        instances.append(tmp)
    return instances


# tok = init_spacy(lower=True)
tok = init_nltk(lower=True)


def postprocess_evaluate(
    path,
    sort_key=None,
    save_to=None,
    cal_metrics=False,
    process_func=postprocess_evaluate_batch,
    item_process_func=None,
):
    """
    for `from allennlp.training.util import evaluate`. post-process the out file for read-friendly.
    """
    try:
        # only one batch
        data = read_json(path)
        instances = process_func(data)
    except JSONDecodeError as e:
        datas = read_jsonl(path)
        instances = []
        for data in datas:
            instances.extend(process_func(data))
    except TypeError:
        instances = data

    if sort_key:
        try:
            instances.sort(key=lambda x: x[sort_key])
        except Exception as e:
            warning(f"ERR: {str(e)}")

    if "output" in instances[0] or "golden_ques" in instances[0]:
        # 归一化
        for item in tqdm(instances, desc="re-tokenize", ncols=100, colour="green"):
            if item_process_func:
                item = item_process_func(item)
            _k = "output" if "output" in item else "golden_ques"
            item["ques_tokens"] = tok(text=item[_k].replace("my question is, ", ""))

            if "predicted_tokens" in item:
                if not item["predicted_tokens"]:
                    item["predicted_tokens"] = [["none"]]
                # print("err", item)
            else:
                # rerank
                item["predicted_tokens"] = tok(text=item["predicted_text"].replace("my question is, ", ""))

            pred = (
                item["predicted_tokens"][0]
                if isinstance(item["predicted_tokens"][0], list)
                else item["predicted_tokens"]
            )
            pred = [""] if not pred else pred
            pred = tok(" ".join(pred).replace("my question is ,", "").strip())
            item["predicted_tokens"] = pred
            try_pop_keys(item, keys=["source_tokens", "pred_tgt", "predicted_tokens_bart"])

    if cal_metrics:
        from rouge import Rouge

        from metrics import cal_BLEU

        rouge = Rouge()
        bleu = cal_BLEU()
        for item in tqdm(instances, desc=f"cal BLEU/ROUGE ing", ncols=100):
            golden = item["ques_tokens"]
            pred = item["predicted_tokens"]
            pred = pred[0] if isinstance(pred[0], list) else pred

            bleu1 = bleu([golden], pred, ngram=1)
            bleu2 = bleu([golden], pred, ngram=2)
            bleu3 = bleu([golden], pred, ngram=3)
            bleu4 = bleu([golden], pred, ngram=4)
            item["bleu-1"] = bleu1
            item["bleu-2"] = bleu2
            item["bleu-3"] = bleu3
            item["bleu-4"] = bleu4

            score = rouge.get_scores(hyps=" ".join(pred), refs=" ".join(golden))[0]
            item["rouge"] = score

            # pop keys
            try_pop_keys(
                item,
                [
                    "pred_probs",
                    "pred_tgt",
                ],
            )
            if "predicted_tokens" in item and isinstance(item["predicted_tokens"][0], list):
                item["predicted_tokens"] = item["predicted_tokens"][:1]

    if save_to:
        save_to_json(instances, save_to)
    return instances


def postprocess_evaluate_for_beam_pmt(predictions_output_file):
    """
    每个item输出多个predict，选择logit最高的一个，计算各种指标
    """
    instances = read_json(predictions_output_file)

    # debug
    # instances = instances[:200]

    for item in instances:
        try_pop_keys(item, keys=["source_tokens", "pred_tgt"])

    from metrics import cal_BLEU

    bleu = cal_BLEU()

    func_max_logit = lambda x: abs(float(x["pred_probs"])) + abs(x["beam_logit"] if "beam_logit" in x else 0)

    func_best = lambda x: bleu(
        golden=[x["ques_tokens"]],
        pred=(
            x["predicted_tokens"][0] if isinstance(x["predicted_tokens"][0], list) else x["predicted_tokens"]
        ),
        ngram=4,
    )

    # max re-rank
    func_max_rerank = lambda x: float(x["pred_score"])

    ins_maxLogit, ins_rerank, ins_Best = [], [], []
    all_ids = sorted(set([i["ID"].split("@beam")[0] for i in instances]))
    for _id in tqdm(all_ids, ncols=100, desc="group sorting"):
        tmp = [i for i in instances if i["ID"].startswith(_id)]
        if tmp and "pred_probs" in tmp[0]:
            beams = sorted(
                tmp,
                key=func_max_logit,
                reverse=False,
            )
            beam = deepcopy(beams[0])
            beam["ID"] = _id
            ins_maxLogit.append(beam)

        beams = sorted(
            tmp,
            key=func_best,
            reverse=True,
        )
        beam = deepcopy(beams[0])
        beam["ID"] = _id
        ins_Best.append(beam)

    from rouge import Rouge

    rouge = Rouge()
    for name, instances in zip(
        ["-MaxLogit.json", "-Best.json"],
        [ins_maxLogit, ins_Best],
    ):
        for item in tqdm(instances, desc=f"cal BLEU/ROUGE ing", ncols=100):
            golden = item["ques_tokens"]
            pred = (
                item["predicted_tokens"][0]
                if isinstance(item["predicted_tokens"][0], list)
                else item["predicted_tokens"]
            )
            pred = [""] if not pred else pred
            bleu1 = bleu([golden], pred, ngram=1)
            bleu2 = bleu([golden], pred, ngram=2)
            bleu3 = bleu([golden], pred, ngram=3)
            bleu4 = bleu([golden], pred, ngram=4)
            item["bleu-1"] = bleu1
            item["bleu-2"] = bleu2
            item["bleu-3"] = bleu3
            item["bleu-4"] = bleu4

            score = rouge.get_scores(hyps=" ".join(pred), refs=" ".join(golden))[0]
            item["rouge"] = score

        save_to = predictions_output_file.replace(".json", name)
        if instances:
            save_to_json(instances, save_to)


def eval_by_group(path, key, vocab=None, exclude_words=None, save_to=None, repeat_dict=None):
    from metrics import METEOR, PaperBLEU, paper_Rouge

    if vocab and isinstance(vocab, Vocab):
        exclude_words = [vocab[vocab.PAD], vocab[vocab.SOS], vocab[vocab.EOS]]
    else:
        exclude_words = exclude_words
    meteor = METEOR(exclude_words=exclude_words)

    datas = read_json(path)

    # debug
    # _ids = set(read_json("bug_test_ids.json"))
    # datas = [i for i in datas if i["ID"] in _ids]
    print(f"test datas len: {len(datas)}")

    # for PQ
    if repeat_dict:
        _id_map = {i["ID"]: index for index, i in enumerate(datas)}
        for k, v in repeat_dict.items():
            if k in _id_map:
                datas.extend([deepcopy(datas[_id_map[k]])] * (v - 1))
        print(f"!! test datas len: {len(datas)}")

    # replace _
    def _c(item):
        item["ques_tokens"] = " ".join(item["ques_tokens"]).replace("_", " ").split(" ")
        item["predicted_tokens"] = (
            item["predicted_tokens"][0]
            if isinstance(item["predicted_tokens"][0], list)
            else item["predicted_tokens"]
        )
        item["predicted_tokens"] = " ".join(item["predicted_tokens"]).replace("_", " ").split(" ")
        return item

    datas = [_c(i) for i in datas]

    for item in datas:
        if key not in item:
            item[key] = "no_value"

    valid_keys = sorted(set([i[key] for i in datas]))
    res_by_group = {k: {} for k in valid_keys}

    def _cal_avg(k, _datas):
        total = len(_datas) + 0.0001
        if k not in res_by_group:
            res_by_group[k] = {}
        # res_by_group[k]["BLEU-4"] = sum([i["bleu-4"] for i in _datas]) / total
        # res_by_group[k]["ROUGE-L-F"] = (
        #     sum([i["rouge"]["rouge-l"]["f"] for i in _datas]) / total
        # )
        predictions = [
            i["predicted_tokens"][0] if isinstance(i["predicted_tokens"][0], list) else i["predicted_tokens"]
            for i in _datas
        ]
        gold_targets = [i["ques_tokens"] for i in _datas]
        # if k == "all":
        meteor.reset()
        meteor(predictions, gold_targets)
        res_by_group[k].update(meteor.get_metric())
        _references = {idx: [" ".join(i).lower()] for idx, i in enumerate(gold_targets)}
        _prediction = {idx: [" ".join(i).lower()] for idx, i in enumerate(predictions)}
        paper_bleu_res = PaperBLEU(4).compute_score(references=_references, prediction=_prediction)
        res_by_group[k].update({"paper_bleu": paper_bleu_res})
        paper_rouge_res = paper_Rouge().compute_score(references=_references, prediction=_prediction)
        res_by_group[k].update({"paper_rouge": paper_rouge_res})

    group_lens = {}
    for k in res_by_group.keys():
        _datas = [i for i in datas if i[key] == k]
        group_lens[k] = len(_datas)
        _cal_avg(k, _datas)

    _cal_avg("all", datas)
    res_by_group["group_by"] = key
    res_by_group["group_lens"] = group_lens

    save_to = os.path.join(os.path.dirname(path), f"eval_groupBy-{key}.json") if save_to is None else save_to
    save_to_json(res_by_group, save_to)

    return res_by_group


if __name__ == "__main__":
    # vocab = Vocab.load_from_file("preprocess/wqcwq_OOV_v1.0/wqcwq_vocab_10000.json")
    res = eval_by_group(
        path="save/wqcwq/graph_to_seq/100%-v01/predictions.json",
        key="compositionality_type",
        # vocab=vocab,
    )
    pprint(res)
