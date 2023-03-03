import functools
from glob import glob
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from common.common_dataclean import clean_html_pipline
from common.common_tokenizers import to_sent
from common.common_utils import read_jsonl, save_to_jsonl

SENT_SPLIT = " @_s!@ "


def to_sent(sp=SENT_SPLIT, return_list=False):
    """
    return_list: return list of str sent.
    """
    import spacy

    nlp = spacy.load("en_core_web_sm")

    def _do(text):
        sents = []
        text_len = 0
        try:
            sents = [[tok.text for tok in sent if tok.text != "\n"] for sent in nlp(text).sents]
            if not return_list:
                text_len = sum([len(sent) for sent in sents])
                sents = sp.join([" ".join(sent) for sent in sents])
            else:
                sents = [" ".join(sent) for sent in sents]
        except Exception as e:
            print(text, e)

        if return_list:
            return sents
        else:
            return text_len, sents

    return _do


ser = to_sent()


def parse_one(path):
    datas = [i for i in read_jsonl(path)]
    new_datas = []
    for item in datas:
        content = clean_html_pipline(item["text"])
        if content:
            text_len, content = ser(content)
            item.pop("text")
            item["content"] = content
            item["tokens_len"] = text_len
            new_datas.append(item)
    path_save = path.replace("enwiki-20181220", "enwiki-20181220-clean") + ".jsonl"
    save_to_jsonl(new_datas, path_save, _print=False)
    return None


def multi_clean_data():
    paths = glob("enwiki-20181220/wiki_raw/**/*")
    pool = Pool(cpu_count() - 5)
    print(f"cpu used: {pool}")
    mapper = functools.partial(parse_one)

    pbar = tqdm(total=len(paths), ncols=80)
    for r in pool.imap(mapper, paths):
        pbar.update(1)
    print("DONE")


if __name__ == "__main__":
    multi_clean_data()
