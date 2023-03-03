import argparse
import gzip
import itertools
import json
import os
import pickle
import random
import re
import subprocess
import sys
import time
from datetime import date, datetime
from glob import glob
from urllib import parse, request

import numpy as np
import torch
import yaml
from tqdm import tqdm

SENT_SPLIT = " @_s!@ "


# I/O
def is_in_notebook():
    return "ipykernel" in sys.modules


def clear_output():
    """
    clear output for both jupyter notebook and the console
    """
    os.system("cls" if os.name == "nt" else "clear")
    if is_in_notebook():
        from IPython.display import clear_output as clear

        clear()


def read_json(path="test.json"):
    with open(path, "r", encoding="utf-8") as f1:
        res = json.load(f1)
    return res


def read_json_from_path(path_patten):
    paths = sorted(glob(path_patten))
    datas = [read_json(i) for i in tqdm(paths, ncols=50, desc=f"Loading from {path_patten}")]
    return datas


def yield_json_from_path(path_patten):
    paths = sorted(glob(path_patten))
    datas = (read_json(i) for i in paths)
    return datas


def read_yaml(path, to_args=False, **kwargs):
    """
    example:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", "-c", default="demo.yml", type=str, help="")
        args = parser.parse_args()
        args = read_yaml(args.config, to_args=True, **vars(args))
        print(args)
    """
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    debug = data["debug"]
    # add
    for k, v in kwargs.items():
        if k in data:
            print(f"overwrite warning! key `{k}` in yml file `{path}` has been replaceed by `{v}` in args.")
        data[k] = v

    # float
    for k in data.keys():
        try:
            if "e" in data[k]:
                data[k] = float(data[k])
        except:
            pass

    if to_args:
        data = argparse.Namespace(**data)

    if not hasattr(data, "distributed") or not data.distributed:
        # torch
        data.device = (
            torch.device(f"cuda:{data.cuda_index}")
            if data.cuda_index > -1 and not debug
            else torch.device("cpu")
        )
        if data.cuda_index > -1 and not debug:
            torch.cuda.set_device(data.cuda_index)

    return data


class ComplexEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(obj, date):
            return obj.strftime("%Y-%m-%d")
        else:
            return json.JSONEncoder.default(self, obj)


def save_to_json(obj, path, _print=True):
    if _print:
        print(f"SAVING: {path}")
    if type(obj) == set:
        obj = list(obj)
    dirname = os.path.dirname(path)
    if dirname and dirname != ".":
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f1:
        json.dump(obj, f1, ensure_ascii=False, indent=4, cls=ComplexEncoder)
    if _print:
        res = subprocess.check_output(f"ls -lh {path}", shell=True).decode(encoding="utf-8")
        print(res)


def read_pkl(path="test.pkl"):
    with open(path, "rb") as f1:
        res = pickle.load(f1)
    return res


def save_to_pkl(obj, path, _print=True):
    dirname = os.path.dirname(path)
    if dirname and dirname != ".":
        os.makedirs(dirname, exist_ok=True)
    with open(path, "wb") as f1:
        pickle.dump(obj, f1)
    if _print:
        res = subprocess.check_output(f"ls -lh {path}", shell=True).decode(encoding="utf-8")
        print(res)


def read_jsonl(path="test.jsonl", desc="", max_instances=None, _id_to_index_key=False):
    with open(path, "r", encoding="utf-8") as f1:
        res = []
        _iter = tqdm(enumerate(f1), desc=desc, ncols=100) if desc else enumerate(f1)
        for idx, line in _iter:
            if max_instances and idx >= max_instances:
                break
            res.append(json.loads(line.strip()))
    if _id_to_index_key:
        id_to_index = {i[_id_to_index_key]: idx for idx, i in enumerate(res)}
        return res, id_to_index
    else:
        return res


def jsonl_generator(path, topn=None, total=None, percent=1, update_func=None, **kwargs):
    """
    usage:
    succ = 0
    def _update():
        return {"success": succ}
    for item in jsonl_generator(path, total=123, update_func=_update):
    """
    total = total if total else file_line_count(path)
    topn = topn if topn else int(percent * total) + 1
    with open(path) as f1:
        pbar = tqdm(f1, total=min(total, topn), ncols=100, **kwargs)
        for idx, line in enumerate(pbar):
            if idx >= topn:
                break
            yield json.loads(line.strip())
            if update_func:
                info = update_func()
                pbar.set_postfix(ordered_dict=info)


def save_to_jsonl(obj, path, _print=True):
    """
    Object of type set is not JSON serializable. so PAY ATTENTION to data type.
    """
    if isinstance(obj, set):
        obj = list(obj)
    elif isinstance(obj, dict):
        obj = obj.items()
    dirname = os.path.dirname(path)
    if dirname and dirname != ".":
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f1:
        for line in obj:
            f1.write(json.dumps(line) + "\n")
    if _print:
        res = subprocess.check_output(f"ls -lh {path}", shell=True).decode(encoding="utf-8")
        print(res)


def save_to_gzip(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = bytes(data, "utf8")
    with gzip.open(path, "wb") as f:
        f.write(data)
    print(f"SAVE: {path}")


def smart_read_line(path):
    """
    support gzip and bz2 format
    tar.gz should be uncompressed
    """
    if path.endswith(".tar.gz"):
        raise ValueError(f"{path} should be uncompressed.")
    elif path.endswith(".gz"):
        import gzip

        with gzip.open(path, "rb") as f:
            for l in f:
                yield (l.decode("utf-8"))
    elif path.endswith(".bz2"):
        import bz2

        with bz2.open(path) as f:
            for l in f:
                yield (l.decode("utf-8"))
    else:
        with open(path, encoding="utf-8") as f:
            for l in f:
                yield (l)


def get_filename(path):
    return os.path.splitext(os.path.basename(path))[0]


# model


def freeze_to_layer_by_name(model, layer_name, exclude_layers=[]):
    if not layer_name:
        return

    # state_dict may not equal to named_parameters.
    keys = [i[0] for i in model.named_parameters()]
    if layer_name == "all":
        index_start = len(keys)
    else:
        index_start = -1
        for index, key in enumerate(keys):
            if layer_name in key:
                index_start = index
                break

    exclude_idxs = set()
    for name in exclude_layers:
        for index, key in enumerate(keys):
            if name in key:
                exclude_idxs.add(index)

    if index_start < 0:
        print(f"Don't find layer name: {layer_name}")
        print(f"must in : \n{keys}")
        return

    grad_nums = 0
    for index, i in enumerate(model.parameters()):
        if index > index_start or index in exclude_idxs:
            i.requires_grad = True
            grad_nums += 1
        else:
            i.requires_grad = False

    print(f"freeze layers num: {index_start}, active layers num: {grad_nums}.")


def get_parameter_number(net, optimizer=None):
    total_num = sum(p.numel() for p in net.parameters())

    if optimizer:
        trainable_num = sum([para.numel() for item in optimizer.param_groups for para in item["params"]])
    else:
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)

    return {"Total": total_num, "Trainable": trainable_num}


# func
def parameters_combinations(paras: dict):
    names = list(paras.keys())
    values = list(paras.values())
    res = list(itertools.product(*values))
    combs = [{names[idx]: v for idx, v in enumerate(vs)} for vs in res]
    return combs


def time_now(fotmat="%Y-%m-%d %H:%M:%S"):
    date_time = datetime.now().strftime(fotmat)
    return date_time


def split_train_dev_test(items, ratio=[0.7, 0.2, 0.1], seed=123):
    assert abs(sum(ratio) - 1.0) < 1e-9
    # ratio.sort()
    # ratio = ratio[::-1]
    random.seed(seed)
    random.shuffle(items)
    if len(ratio) == 2:
        _num = int(len(items) * ratio[0])
        return items[:_num], items[_num:]
    elif len(ratio) == 3:
        _num1 = int(len(items) * ratio[0])
        _num2 = int(len(items) * (ratio[0] + ratio[1]))
        return items[:_num1], items[_num1:_num2], items[_num2:]
    else:
        raise ValueError("ratio length is not correct")


def tokenized_to_device(res_dict, DEVICE):
    res_dict = {k: v.to(DEVICE) for k, v in res_dict.items()}
    return res_dict


def _safe_division(numerator, denominator):
    if abs(denominator) < 1e-9:
        return 0
    return numerator / denominator


def try_pop_keys(item, keys):
    for k in keys:
        if k in item:
            item.pop(k)
    return item


def get_logger(path="logs.log", logger_name=__name__, filemode="a"):
    """
    usage:
        logger = get_logger()
        logger.info()
    """
    import logging

    logging.basicConfig(
        level=logging.WARN,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename=path,
        filemode=filemode,
        force=True,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger = logging.getLogger(logger_name)
    return logger


# pandas


def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                # numpy.iinfo() shows the machine limits for integer types.
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print("memory usage after optimization is: {:.2f} MB".format(end_mem))
        print("decrease by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    return df


# html
def replace_ignorecase(text, _from, _to):
    try:
        res = re.sub(str(_from), str(_to), str(text), flags=re.I)
    except Exception as e:
        # print(f"\nerror from `{_from}` to `{_to}`.")
        res = text.replace(_from, _to)
    return res


def replaces(text, maps, ignorecase=False):
    for k, v in maps.items():
        if ignorecase:
            text = replace_ignorecase(text, k, v)
        else:
            text = text.replace(k, v)
    return text


def colorful(text, color="yellow"):
    if color == "yellow":
        text = "\033[1;33;33m" + str(text) + "\033[0m"
    elif color == "grey":
        text = "\033[1;30;47m" + str(text) + "\033[0m"
    elif color == "green":
        text = "\033[1;32;32m" + str(text) + "\033[0m"
    elif color == "red":
        text = "\033[1;31;40m" + str(text) + "\033[0m"
    else:
        pass
    return text


# demo
def make_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="data", type=str, help="data path")
    parser.add_argument("--load_model", action="store_true", help="whether to load")
    parser.set_defaults(load_model=True)

    args = parser.parse_args()
    return args


# 多参数，多进程


def _parse_one(_tuple):
    x, y_list = _tuple
    for i in y_list:
        i["ttt"] = x
    return {x: y_list}


def multi_demo():
    import functools
    from multiprocessing import Pool, cpu_count

    pool = Pool(cpu_count())
    mapper = functools.partial(_parse_one)
    inputs = {
        "asd1": [{"a": 1230}],
        "asd2": [{"a": 1230}],
        "asd3": [{"a": 1230}],
        "asd4": [{"a": 1230}],
    }.items()
    pbar = tqdm(total=len(inputs), ncols=80)
    res = []
    for r in pool.imap(mapper, inputs):
        res.append(r)
        pbar.update(1)
        pbar.set_description_str(f"qweqwe")
        pbar.set_postfix(ordered_dict={"k": 1})
        pbar.set_postfix_str(s="")
    print(res)


def _tokenize_to_file(args, i, num_process, in_path, out_path, line_fn):

    with open(in_path, "r", encoding="utf-8") as in_f, open("{}_split{}".format(out_path, i), "wb") as out_f:
        for idx, line in enumerate(in_f):
            if idx % num_process != i:
                continue
            out_f.write(line_fn(args, line, tokenizer))


def multi_demo_2(args, num_process=8):
    from multiprocessing import Process

    processes = []
    for i in range(num_process):
        p = Process(
            target=_tokenize_to_file,
            args=(args, i, num_process, in_path, out_path, line_fn),
        )
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


# wrapper


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        print("time consume: {:.2f}s .".format(time.time() - start))
        return res

    return wrapper


import functools
from concurrent import futures


def timeout(seconds):
    executor = futures.ThreadPoolExecutor(1)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            future = executor.submit(func, *args, **kw)
            return future.result(timeout=seconds)

        return wrapper

    return decorator


def wc_l(path):
    try:
        res = subprocess.check_output(f"wc -l {path}", shell=True).decode(encoding="utf-8")
        line_num = int(res.split()[0])
    except Exception as e:
        line_num = None
    return line_num


# @timeout(10)
def file_line_count(path):
    return wc_l(path)

