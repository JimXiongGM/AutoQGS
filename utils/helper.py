import os
import re

from common.common_utils import get_logger, replaces
from data_helper.freebase_wrapper import FreebaseWrapper

REPLACES = {
    "(": " ( ",
    ")": " ) ",
    "{": " { ",
    "}": " } ",
    "\n": " ",
    "\t": " ",
}


def _add_edge_symbol(sparql):
    """
    Args:
        sparql (str)

    Returns:
        str
    """
    for res in sorted(set(re.findall(" (ns:.*?) ", sparql))):
        if res.count(".") > 1:
            # _res = re.sub("[_\.]", " ", res)
            # new_res = f"_e1_ {_res} _e2_"
            new_res = res.replace("_", " _ ")
            sparql = sparql.replace(res, new_res)
    return sparql


def replace_mid_in_sparql(sparql, fb: FreebaseWrapper, data_name=None, _nltk=False):
    _patt1 = "(ns:[mg]\..*?)[ \\n\\t)]"
    _patt2 = "ns:"
    # sparql = _add_edge_symbol(sparql)
    mids = set(re.findall(_patt1, sparql))
    for mid in mids:
        name = fb.get_mid_name(mid.replace(_patt2, ""))
        name = name or "no_name"
        if _nltk:
            name = name.replace('"', " ")
        sparql = sparql.replace(mid, _patt2 + f'"{name}"')

    if _nltk:
        REPLACES["_"] = " "
        REPLACES["?"] = "_var_"
        REPLACES["."] = " . "
        REPLACES["<"] = " < "
        REPLACES["/"] = " / "
        REPLACES[":"] = " : "
        REPLACES['"'] = ' " '
        REPLACES["'"] = ' " '

    sparql = re.sub(" +", " ", replaces(sparql, REPLACES).strip())
    return sparql


def format_lambda(logical):
    logical = re.sub(" +", " ", replaces(logical, REPLACES).strip())
    return logical


def combine_paths(args):
    # debug
    if args.debug:
        args.num_epochs = 1
        args.max_instances_train = 9
        args.train_batch_size = 5
        args.max_instances_dev = 9
        args.dev_batch_size = 5
        args.max_instances_test = 9
        args.test_batch_size = 5
        args.ver = "debug-" + args.ver
        if hasattr(args, "train_dataset"):
            args.train_dataset = args.dev_dataset if hasattr(args, "dev_dataset") else args.train_dataset
        args.max_instances_percent_train = 0.001
        args.max_instances_percent_dev = 0.01
        args.dev_batches_per_epoch = 3

    args.serialization_dir = os.path.join(args.serialization_dir, args.ver)
    os.makedirs(args.serialization_dir, exist_ok=True)
    args.predictions_output_file = os.path.join(args.serialization_dir, args.predictions_output_file)
    args.vocabulary = os.path.join(args.serialization_dir, args.vocabulary)
    args.load_weights_file = os.path.join(args.serialization_dir, args.load_weights_file)
    args.save_weights_file = os.path.join(args.serialization_dir, args.save_weights_file)
    args.metrics_output_file = os.path.join(args.serialization_dir, args.metrics_output_file)
    args.log_name = os.path.join(args.serialization_dir, args.log_name)

    # logger
    logger_path = os.path.join(args.serialization_dir, "log.log")
    args.logger = get_logger(path=logger_path, logger_name=args.ver)

    if not hasattr(args, "grid_search"):
        args.grid_search = False

    return args
