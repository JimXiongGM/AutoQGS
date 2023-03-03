import functools
import glob
from collections import Counter
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from common.common_algorithms import Jaccard_coefficient
from common.common_tokenizers import init_spacy
from common.common_utils import read_json, replaces, save_to_json

tokenize = init_spacy(lower=True)
tokenize(text="hello, this. is good")


def _parse_one(item):
    item["tokens"] = tokenize(text=item["question"])
    return item


def pre_tokenized():
    paths = glob.glob("datasets/PathQuestion/*.txt")
    data = []
    for path in paths:
        name = path.split("/")[-1].split(".")[0]
        with open(path) as f:
            for idx, line in enumerate(f):
                q = line.strip().split("\t")[0]
                data.append({"ID": f"{name}@{idx}", "question": q})

    # debug
    # for i in _gen():
    #     y = _parse_one(i)

    pool = Pool(cpu_count())
    mapper = functools.partial(_parse_one)

    res = []
    for r in pool.imap(mapper, data):
        res.append(r)

    save_to_json(res, "preprocess_bart/PathQuestion/pre_tokens.json")


# step 2
my_data = read_json("preprocess_bart/PathQuestion/pre_tokens.json")


def _match(item):
    idx = item["tgt_idx"]
    q = item["tgt_q"].split(" ")
    jaccards = [(idx, i["ID"], Jaccard_coefficient(q, i["tokens"]), i["tokens"]) for i in my_data]
    jaccards.sort(key=lambda x: x[2], reverse=True)
    res = jaccards[0]
    item["match_ID"] = res[1]
    item["match_jaccard"] = res[2]
    item["match_ques"] = res[3]

    return item


def align_with_jointGT(name="test"):
    # Download from jointGT github: https://github.com/thu-coai/JointGT
    tgt_datas = read_json(f"datasets/from_jointGT/pq/{name}.json")
    _map = {
        "`": "",
        " ": "",
        "?": "",
        "'": "",
        '"': "",
        ",": "",
        ".": "",
        "_": "",
        "-": "",
    }
    res = []
    my_map = {replaces(i["question"].lower(), _map): i["ID"] for i in my_data}

    def _gen():
        for idx, item in enumerate(tqdm(tgt_datas, ncols=100)):
            q = item["text"][0].lower()
            _q = replaces(q, _map)
            tmp = {}
            tmp["tgt_idx"] = idx
            if _q not in my_map:
                tmp["tgt_q"] = q
                yield tmp
            else:
                # 进不来
                tmp["match_ID"] = my_map[_q]
                res.append(tmp)

    # debug
    for i in _gen():
        res.append(_match(i))

    # pool = Pool(cpu_count())
    # mapper = functools.partial(_match)
    res2 = []
    # for r in pool.imap(mapper, _gen()):
    #     res2.append(r)

    nogoods = [i for i in res2 if i["match_jaccard"] < 0.9]
    for i in nogoods:
        i["match_ques"] = " ".join(i["match_ques"])
    nogoods.sort(key=lambda x: x["match_jaccard"])
    save_to_json(res + res2, f"preprocess_bart/PathQuestion/match_res_{name}.json")
    print(f"{name} no match len: {len(nogoods)}")
    save_to_json(nogoods, f"preprocess_bart/PathQuestion/nogoods_{name}.json")


def re_oucnt(name="test"):
    match_datas = read_json(f"preprocess_bart/PathQuestion/match_res_{name}.json")
    c = Counter([i["match_ID"] for i in match_datas])
    repeats = dict([i for i in c.most_common() if i[1] > 1])
    save_to_json(repeats, f"datasets/PathQuestion-all-infos-v1.0/test_ids_repeats.json")


if __name__ == "__main__":
    # pre
    pre_tokenized()

    # align
    align_with_jointGT("test")
    align_with_jointGT("dev")

    re_oucnt(name="test")
