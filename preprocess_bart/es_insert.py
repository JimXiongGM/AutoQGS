import glob
import time

from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

from common.common_utils import read_jsonl


def set_mappings(es: Elasticsearch, index):
    """
    ['id', 'revid', 'url', 'title', 'content']

    - id
        id

    - text
        url
        title
        content

    - keyword
        revid
    """
    my_mappings = {
        "settings": {"number_of_shards": 2, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                # keyword
                "revid": {"type": "keyword"},
                # text
                "url": {"type": "text"},
                "title": {
                    "type": "text",
                },
                "content": {
                    "type": "text",
                },
            }
        },
    }

    creat_index_response = es.indices.create(index=index, ignore=400, **my_mappings)
    return creat_index_response


def bulk_insert(es: Elasticsearch, actions):
    try:
        for success, info in helpers.parallel_bulk(es, actions, chunk_size=5000, thread_count=16):
            return info
    except:
        time.sleep(10)
        res = helpers.bulk(es, actions, chunk_size=5000)
    return res


def refresh(es, index):
    res = es.indices.refresh(index=index)
    return res


def delete_index(es, index):
    res = es.indices.delete(index=index, ignore=[400, 404])
    return res


def _parse(path, index, _idkey):
    datas = read_jsonl(path)
    for item in datas:
        if item["content"]:
            item["_index"] = index
            item["_id"] = item.pop(_idkey)
            # print(content,"\n\n")
            yield item


es = Elasticsearch(hosts=["localhost:9200"], timeout=30, max_retries=10, retry_on_timeout=True)


def run_insert(index, _idkey):
    delete_index(es, index)
    set_mappings(es, index=index)
    _dir = "enwiki-20181220-clean/wiki_raw/**/wiki_*"
    paths = glob.glob(_dir)[:]
    paths.sort()
    pbar = tqdm(paths, total=len(paths), ncols=100, colour="green")
    acc_num = 0
    actions = []
    for path in pbar:
        for item in _parse(
            path=path,
            index=index,
            _idkey=_idkey,
        ):
            actions.append(item)
            try:
                if len(actions) > 5000:
                    res = bulk_insert(es, actions)
                    # print(res)
                    acc_num += 5000
                    actions = []
                    pbar.set_description_str(f"accu items: {acc_num}")
            except Exception as e:
                refresh(es, index)
                print(e)
                break
        pbar.update()

    res = bulk_insert(es, actions)
    actions = []
    refresh(es, index)
    print("DONE")


index = "wikipedia2018_v1.1"


def search_test():
    dsl = {"query": {"match": {"content": "hello world"}}}
    result = es.search(index=index, **dsl)
    print(result)


if __name__ == "__main__":
    run_insert(index, _idkey="id")
    search_test()
