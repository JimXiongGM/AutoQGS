import argparse
import os
import tempfile
import warnings

import torch
import torch.optim as optim
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer
from allennlp.training.learning_rate_schedulers import LinearWithWarmup
from tqdm import tqdm

from common.common_utils import colorful, parameters_combinations, read_json, read_yaml, save_to_json
from utils import combine_paths, eval_by_group, myEpochCallback, set_seed
from utils.allen_utils import evaluate
from utils.file_process import postprocess_evaluate, postprocess_evaluate_for_beam_pmt

warnings.filterwarnings("ignore")

from model.s2s_bart_official import Bart

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class s2sEpochCallback(myEpochCallback):
    def __init__(self, reader=None, args=None):
        super().__init__(reader=reader, args=args)

    def on_batch(self, *k, **kwagrs):
        pass


# -----------------------------model----------------------------- #


def get_reader(args, max_instances=None, max_instances_percent=None, top_beam=None, mode="test"):
    if args.dataset in ["wqcwq", "grailQA"]:
        from reader.ds_bart import DSDatasetReader as Reader
    elif args.dataset == "simpleq":
        from reader.ds_bart import SimpleQuestionsReader as Reader
    elif args.dataset == "pathq":
        from reader.ds_bart import PathQuestionsReader as Reader
    else:
        raise ValueError(f"err dataset: {args.dataset}")

    reader = Reader(
        model_name=args.model_name,
        max_instances=max_instances,
        setting=args.setting,
        max_instances_percent=max_instances_percent,
        top_beam=top_beam or args.top_beam if hasattr(args, "top_beam") else None,
        decoder_instructions=args.decoder_instructions,
        beam_sample=args.beam_sample,
        beam_score_map=args.beam_score_map if hasattr(args, "beam_score_map") else None,
        reward_map=args.reward_map if hasattr(args, "reward_map") else None,
        ids_map=args.ids_map,
        max_length=args.max_length if hasattr(args, "max_length") else 512,
        mode=mode,
        only_constrain=args.only_constrain,
        aug_mode=args.aug_mode if hasattr(args, "aug_mode") else None,
    )

    return reader


def get_model(vocab, args):
    model = Bart(
        vocab=vocab,
        model_name=args.model_name,
        max_decoding_steps=args.max_decoding_steps,
        decoder_prompt=args.decoder_prompt,
        decoder_instructions=args.decoder_instructions,
        beam_length_penalty=args.beam_length_penalty,
        beam_size=args.beam_size,
        reward_mode=bool(args.reward_map) if hasattr(args, "reward_map") else False,
    ).to(args.device)
    return model


# -----------------------------main----------------------------- #


def train(args):
    vocab = Vocabulary.from_pretrained_transformer(model_name=args.model_name)

    reader_train = get_reader(
        args,
        max_instances_percent=args.max_instances_percent_train,
        max_instances=args.max_instances_train,
        mode="train",
    )
    reader_dev = get_reader(
        args,
        max_instances_percent=args.max_instances_percent_dev,
        max_instances=args.max_instances_dev,
        mode="dev",
    )

    # dataloader
    num_workers = 0
    train_loader = MultiProcessDataLoader(
        reader=reader_train,
        data_path=args.datasets,
        batch_size=args.train_batch_size,
        drop_last=False,
        shuffle=True,
        batch_sampler=None,
        batches_per_epoch=None,
        num_workers=num_workers,
        max_instances_in_memory=None,
        start_method="fork",
        cuda_device=args.device,
    )
    dev_loader = MultiProcessDataLoader(
        reader=reader_dev,
        data_path=args.datasets,
        batch_size=args.dev_batch_size,
        drop_last=False,
        shuffle=False,
        batch_sampler=None,
        batches_per_epoch=args.dev_batches_per_epoch,
        num_workers=num_workers,
        max_instances_in_memory=None,
        start_method="fork",
        cuda_device=args.device,
    )
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)

    args.train_datas_len = len(train_loader._instances)
    args.dev_datas_len = len(dev_loader._instances)

    print(colorful(f"train len: {args.train_datas_len}"))
    print(colorful(f"dev len: {args.dev_datas_len}"))

    # model
    model = get_model(vocab, args)
    if args.load_model:
        if isinstance(args.load_model, str):
            _load_weights_file = args.load_model
        else:
            _load_weights_file = args.load_weights_file
        model.load_state_dict(torch.load(_load_weights_file, map_location=args.device))
    if hasattr(args, "init_from") and args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location=args.device))

    # for bart
    no_decay = ["bias", "LayerNorm.weight"]
    bart_param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in bart_param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate,
        },
        {
            "params": [p for n, p in bart_param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.learning_rate,
        },
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    len_train = len(train_loader._instances)
    num_steps_per_epoch = len_train // (args.train_batch_size * args.num_gradient_accumulation_steps)
    # t_total = num_steps_per_epoch * args.num_epochs
    t_total = max(num_steps_per_epoch * min(args.num_epochs, 30), 50)
    args.warmup_total_steps = int(t_total)
    args.warmup_steps = max(int(t_total * args.warmup_proportion), 5)
    print(colorful(text=f"Total steps: {args.warmup_total_steps}  warm steps: {args.warmup_steps}"))

    learning_rate_scheduler = LinearWithWarmup(
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        num_steps_per_epoch=num_steps_per_epoch,
        warmup_steps=args.warmup_steps,
    )

    with tempfile.TemporaryDirectory() as serialization_dir:
        epoch_callbacks = [s2sEpochCallback(args=args)]

        trainer = GradientDescentTrainer(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            patience=args.patience_trainer,
            validation_metric="+BLEU-4",
            validation_data_loader=dev_loader,
            num_epochs=args.num_epochs,
            serialization_dir=serialization_dir,
            checkpointer=None,
            cuda_device=args.device,
            grad_norm=True,
            grad_clipping=args.grad_clipping,
            learning_rate_scheduler=learning_rate_scheduler,
            momentum_scheduler=None,
            moving_average=None,
            callbacks=epoch_callbacks,
            distributed=False,
            local_rank=0,
            world_size=1,
            num_gradient_accumulation_steps=args.num_gradient_accumulation_steps,
            use_amp=args.device != torch.device("cpu"),
            # use_amp=False,
            run_confidence_checks=False,
            skip_e=10 if args.max_instances_percent_train < 0.1 else -1,
        )
        trainer.train()
    print(colorful("train DONE", color="red"))


def test(args):
    # _top_beam = 10 if hasattr(args, "reward_map") and args.reward_map else args.top_beam
    if args.eval_all:
        vocab = Vocabulary.from_pretrained_transformer(model_name=args.model_name)
        reader = get_reader(args, max_instances=args.max_instances_test, mode="test")
        model = get_model(vocab, args)
        if args.update_layers is not None:
            if isinstance(args.load_model, str):
                _load_weights_file = args.load_model
            else:
                _load_weights_file = args.load_weights_file
            model.load_state_dict(torch.load(_load_weights_file, map_location=args.device))
        test_loader = MultiProcessDataLoader(
            reader=reader,
            data_path=args.datasets,
            batch_size=args.test_batch_size,
            drop_last=False,
            shuffle=False,
            batch_sampler=None,
            batches_per_epoch=(
                args.test_batches_per_epoch if hasattr(args, "test_batches_per_epoch") else None
            ),
            num_workers=0,
            max_instances_in_memory=128,
            start_method="fork",
            cuda_device=args.device,
        )
        test_loader.index_with(vocab)
        # args.test_datas_len = len(test_loader._instances)

        # print(colorful(f"test len: {args.test_datas_len}"))

        evaluate(
            model=model,
            data_loader=test_loader,
            cuda_device=args.device,
            batch_weight_key=None,
            output_file=args.metrics_output_file,
            predictions_output_file=args.predictions_output_file,
            total=334834 // args.test_batch_size,
        )

    if args.setting in [
        "only-autoprompt",
        "sparql-autoprompt",
        "sparql-middesc",
        "middesc-autoprompt",
        "sparql-middesc-autoprompt",
    ]:
        re_eval_test(args)
    else:
        postprocess_evaluate(
            path=args.predictions_output_file,
            sort_key="ID",
            save_to=args.predictions_output_file,
            cal_metrics=False,
        )
        repeat_dict = (
            read_json("datasets/PathQuestion-all-infos-v1.0/test_ids_repeats.json")
            if args.dataset == "pathq"
            else None
        )
        _k = "level"  # compositionality_type level
        eval_by_group(
            path=args.predictions_output_file,
            key=_k,
            repeat_dict=repeat_dict,
        )


def re_eval_test(args):
    """
    Re-calculate test results
    strategy in ["max_logit","max_len","best"]
    """
    _k = "level"  # compositionality_type level
    postprocess_evaluate(
        path=args.predictions_output_file,
        sort_key="ID",
        save_to=args.predictions_output_file,
        cal_metrics=False,
    )
    postprocess_evaluate_for_beam_pmt(predictions_output_file=args.predictions_output_file)
    # max logit
    beam_maxlogit = args.predictions_output_file.replace(".json", "-MaxLogit.json")
    repeat_dict = (
        read_json("datasets/PathQuestion-all-infos-v1.0/test_ids_repeats.json")
        if args.dataset == "pathq"
        else None
    )
    eval_by_group(
        path=beam_maxlogit,
        key=_k,
        save_to=os.path.join(
            os.path.dirname(beam_maxlogit),
            f"eval-MaxLogit-groupBy-{_k}.json",
        ),
        repeat_dict=repeat_dict,
    )


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config/baseline.yml", type=str, help="")
    args = parser.parse_args()
    args = read_yaml(args.config, to_args=True, **vars(args))
    args = combine_paths(args)
    return args


def run(args):
    if hasattr(args, "re_eval_test") and args.re_eval_test:
        re_eval_test(args)
    elif args.only_test_mode:
        test(args)
    else:
        # baseline lstm
        train(args)
        if args.test_mode:
            test(args)


def grid_search(args):
    _f = "pq_per100_paras_combinations.json"
    args.patience_scheduler = 2
    args.patience_trainer = 10
    spaces = {
        "learning_rate": [5e-5, 3e-5],
        "train_batch_size": [24, 32],
        "beam_length_penalty": [1.0, 3.0],
        "beam_size": [5],
        "warmup_proportion": [0.08, 0.16],
    }
    combs = parameters_combinations(spaces)
    save_to_json(combs, _f)

    combs = read_json(_f)
    _dir = args.serialization_dir + "-paras"
    for i, comb in enumerate(tqdm(combs, ncols=100)):
        args.serialization_dir = f"{_dir}/{i}"
        args.predictions_output_file = f"{args.serialization_dir}/predictions.json"
        args.metrics_output_file = f"{args.serialization_dir}/evaluate_metric.json"
        args.log_name = f"{args.serialization_dir}/log.json"

        os.makedirs(args.serialization_dir, exist_ok=True)
        args.save_weights_file = args.load_weights_file = f"{args.serialization_dir}/weight.th"
        if os.path.exists(f"{args.serialization_dir}/eval-MaxLogit-groupBy-compositionality_type.json"):
            continue
        print(colorful(f"current: {i}", color="red"))
        print(comb)
        for k, v in comb.items():
            setattr(args, k, v)
        # debug
        # args.num_epochs = 1
        train(args)
        test(args)
        os.remove(args.load_weights_file)


if __name__ == "__main__":
    args = make_args()
    set_seed()
    if not hasattr(args, "only_constrain"):
        args.only_constrain = None

    if args.grid_search:
        grid_search(args)
    else:
        run(args)
