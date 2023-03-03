import argparse
import functools
import os
import warnings
from multiprocessing import Pool

import torch
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.vocabulary import Vocabulary

from common.common_utils import try_pop_keys
from utils import eval_by_group, set_seed
from utils.allen_utils import evaluate
from utils.file_process import postprocess_evaluate, postprocess_evaluate_for_beam_pmt

warnings.filterwarnings("ignore")

from model.s2s_bart_official import Bart

# raw
# from allennlp_models.generation.models import Bart
from reader.ds_bart import DSDatasetReader

os.environ["TOKENIZERS_PARALLELISM"] = "false"


"""
Rerank the output of pmt
1. Predict results for train 10% dev test data, calculate bleu-4 and rougel between output and answer
"""

model_name = "pretrained_models/facebook/bart-base"
top_beam = 10


def get_reader(args, max_instances=None, max_instances_percent=None):
    reader = DSDatasetReader(
        model_name=model_name,
        max_instances=max_instances,
        setting=args.setting,
        max_instances_percent=max_instances_percent,
        top_beam=top_beam,
    )
    return reader


def get_model(vocab):
    model = Bart(
        vocab=vocab,
        model_name=model_name,
        max_decoding_steps=512,
    )
    return model


os.makedirs("datasets/for_rerank/", exist_ok=True)


def predict(args):
    vocab = Vocabulary.from_pretrained_transformer(model_name=model_name)
    # pbar = tqdm(total=18, ncols=100, colour="blue")
    # for percent in [0.1, 0.5, 1, 5, 10, 100]:
    for percent in [100]:
        os.makedirs(f"datasets/for_rerank/{args.setting}-beam{top_beam}-{percent}%", exist_ok=True)
        # save/s2s-bart-fewshot/sparql-middesc-autoprompt/top50-sample-100%-all-v01/weights.th
        weight_file = f"save/s2s-bart-fewshot/{args.setting}/top50-sample-{percent}%-all-v01/weights.th"
        model = get_model(vocab).to(args.device)
        model.load_state_dict(torch.load(weight_file, map_location=args.device))

        for mode in [
            "train",
            "dev",
            "test",
        ]:
            # pbar.set_description_str(f"doing {percent}%-{mode}")
            if mode == "train":
                reader = get_reader(args, max_instances_percent=percent / 100, max_instances=None)
                data_path = args.train_dataset
                predictions_output_file = (
                    f"datasets/for_rerank/{args.setting}-beam{top_beam}-{percent}%/predictions-train.json"
                )

            elif mode == "dev":
                reader = get_reader(args, max_instances_percent=1, max_instances=None)
                data_path = args.dev_dataset
                predictions_output_file = (
                    f"datasets/for_rerank/{args.setting}-beam{top_beam}-{percent}%/predictions-dev.json"
                )

            else:
                reader = get_reader(args, max_instances=None)
                data_path = args.test_dataset
                predictions_output_file = (
                    f"datasets/for_rerank/{args.setting}-beam{top_beam}-{percent}%/predictions-test.json"
                )

            max_instances_in_memory = None
            batch_size = 4
            data_loader = MultiProcessDataLoader(
                reader=reader,
                data_path=data_path,
                batch_size=batch_size,
                drop_last=False,
                shuffle=False,
                batch_sampler=None,
                batches_per_epoch=None,
                num_workers=0,
                max_instances_in_memory=max_instances_in_memory,
                start_method="fork",
                cuda_device=args.device,
            )
            data_loader.index_with(vocab)

            evaluate(
                model=model,
                data_loader=data_loader,
                cuda_device=args.device,
                batch_weight_key=None,
                output_file=None,
                predictions_output_file=predictions_output_file,
                total=len(data_loader._instances) // batch_size if max_instances_in_memory is None else None,
            )
            postprocess_evaluate(
                path=predictions_output_file,
                sort_key="ID",
                save_to=predictions_output_file,
                cal_metrics=True,
                item_process_func=_post_process,
            )
            # pbar.update()


def _post_process(item):
    try_pop_keys(
        item,
        [
            "source_tokens",
            "pred_tgt",
        ],
    )
    item["input"] = item["input"].replace("my question is,", "").strip()
    return item


def re_eval_test(args):
    key = "compositionality_type"
    predictions_output_file = f"datasets/for_rerank/{args.setting}-beam{top_beam}-100%/predictions-test.json"
    postprocess_evaluate_for_beam_pmt(predictions_output_file=predictions_output_file)

    # best
    beam_best = predictions_output_file.replace(".json", "-Best.json")
    eval_by_group(
        path=beam_best,
        key=key,
        save_to=os.path.join(
            os.path.dirname(beam_best),
            f"eval-Best-groupBy-{key}.json",
        ),
    )


args = {}
args["train_dataset"] = "datasets/for_pmt/pmt-all-desc-final_NLs-Beam_top100-2.1/86-88-train.json"
args["dev_dataset"] = "datasets/for_pmt/pmt-all-desc-final_NLs-Beam_top100-2.1/86-88-dev.json"
args["test_dataset"] = "datasets/for_pmt/pmt-all-desc-final_NLs-Beam_top100-2.1/86-88-test.json"

args["device"] = torch.device("cuda:0")


# middesc-autoprompt sparql-middesc-autoprompt
args["setting"] = "middesc-autoprompt"


args = argparse.Namespace(**args)
set_seed()


def multi_predict():
    modes = [("train", 0), ("dev", 1), ("test", 2)]
    pool = Pool(3)
    mapper = functools.partial(predict, args)
    for r in pool.imap(mapper, modes):
        pass

