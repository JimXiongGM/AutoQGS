import argparse
import os
import tempfile
import warnings
from copy import deepcopy

import torch
import torch.optim as optim
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer
from allennlp.training.learning_rate_schedulers import LinearWithWarmup
from tqdm import tqdm

from common.common_utils import (
    colorful,
    file_line_count,
    read_json,
    read_yaml,
    save_to_jsonl,
    time_now,
)
from utils import combine_paths, myEpochCallback, set_seed
from utils.allen_utils import evaluate
from utils.file_process import postprocess_evaluate

warnings.filterwarnings("ignore")


# raw
# from allennlp_models.generation.models import Bart
from model.ds_bart import DSBart

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class s2sEpochCallback(myEpochCallback):
    def __init__(self, args=None):
        super().__init__(args=args)

    def on_batch(self, *k, **kwagrs):
        pass

    def on_epoch(
        self,
        trainer: "GradientDescentTrainer",
        metrics,
        epoch: int,
        is_primary: bool,
    ):
        if is_primary:
            print(colorful(text=f"Epoch: {epoch} DONE!", color="yellow"))
            if type(self.logs) == list and epoch >= 0:
                metrics["save_time"] = time_now()
                self.logs.append(deepcopy(metrics))

            # save every epoch!
            self.save_model(model=trainer.model, ver=epoch)
            self.best_epoch = metrics["best_epoch"]
            self.save_logs(model=trainer.model, optimizer=trainer.optimizer)


# -----------------------------model----------------------------- #


def get_reader(args, max_instances=None, mode="train"):
    max_instances_percent = args.max_instances_percent if hasattr(args, "max_instances_percent") else None

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
        mode=mode,
        # rerun_ids=args.rerun_ids if hasattr(args, "rerun_ids") else None,
        max_length=args.max_length if hasattr(args, "max_length") else 512,
        only_constrain=args.only_constrain if hasattr(args, "only_constrain") else None,
    )
    return reader


def get_model(vocab, args):
    args.beam_size = 50 if hasattr(args, "rerun_ids") and os.path.exists(args.rerun_ids) else args.beam_size
    model = DSBart(
        vocab=vocab,
        model_name=args.model_name,
        logger=args.logger,
        beam_size=args.beam_size,
        max_decoding_steps=args.max_decoding_steps,
        beam_length_penalty=args.beam_length_penalty,
    ).to(args.device)
    return model


# -----------------------------main----------------------------- #


def train(args):
    vocab = Vocabulary.from_pretrained_transformer(model_name=args.model_name)
    reader_train = get_reader(args, max_instances=args.max_instances_train, mode="train")

    # dataloader
    num_workers = 0
    train_loader = MultiProcessDataLoader(
        reader=reader_train,
        data_path=args.train_dataset,
        batch_size=args.train_batch_size,
        drop_last=False,
        shuffle=False,
        batch_sampler=None,
        batches_per_epoch=None,
        num_workers=num_workers,
        max_instances_in_memory=500,
        start_method="fork",
        cuda_device=args.device,
    )
    train_loader.index_with(vocab)

    # model
    model = get_model(vocab, args)
    if args.load_model:
        model.model.load_state_dict(torch.load(args.init_from, map_location=args.device))

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
    len_train = file_line_count(args.train_dataset)
    num_steps_per_epoch = len_train // args.train_batch_size // args.num_gradient_accumulation_steps
    # t_total = num_steps_per_epoch * args.num_epochs
    t_total = num_steps_per_epoch * min(args.num_epochs, 20)
    args.warmup_steps = int(t_total * args.warmup_proportion)

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
            validation_metric="-training_loss",
            validation_data_loader=None,
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
        )
        trainer.train()


def _beam_docode(args, path):
    """
    batch decode predictions indeies to text tokens.
    """
    datas = read_json(path)
    from transformers import BartTokenizerFast

    tok = BartTokenizerFast.from_pretrained(args.model_name)

    def _p(item):
        if "source_tokens" in item:
            item.pop("source_tokens")
        # item.pop("input")
        item["predictions"] = tok.batch_decode(item["predictions"], skip_special_tokens=True)
        return item

    datas = [_p(i) for i in tqdm(datas, ncols=100, desc="decoding")]
    os.remove(path)
    save_to_jsonl(datas, path.replace(".json", ".jsonl"))


def predict(args):
    vocab = Vocabulary.from_pretrained_transformer(model_name=args.model_name)
    reader = get_reader(args, max_instances=args.max_instances_test, mode="test")

    re_beam_mode = True if hasattr(args, "rerun_ids") and os.path.exists(args.rerun_ids) else False
    if re_beam_mode:
        print(colorful(f"mode: re-run beam search!"))

    model = get_model(vocab, args)
    for ver in args.predict_vers:
        # dataloader
        num_workers = 0
        loader = MultiProcessDataLoader(
            reader=reader,
            data_path=args.dataset_file,
            batch_size=args.test_batch_size if not re_beam_mode else 8,
            drop_last=False,
            shuffle=False,
            batch_sampler=None,
            batches_per_epoch=None,
            num_workers=num_workers,
            max_instances_in_memory=100,
            start_method="fork",
            cuda_device=args.device,
        )
        loader.index_with(vocab)

        model.load_state_dict(torch.load(args.save_weights_file + f"-{ver}", map_location=args.device))
        if re_beam_mode:
            _p = args.predictions_output_file.replace(
                ".json",
                f"-{args.dataset}-ver-{ver}-penalty-{args.beam_length_penalty}-err-rerun.json",
            )
        else:
            _p = args.predictions_output_file.replace(
                ".json",
                f"-{args.dataset}-ver-{ver}-penalty-{args.beam_length_penalty}.json",
            )
        evaluate(
            model=model,
            data_loader=loader,
            cuda_device=args.device,
            batch_weight_key=None,
            output_file=None,
            predictions_output_file=_p,
            # total=len(loader._instances)
            # // (args.test_batch_size if not re_beam_mode else 8),
        )
        postprocess_evaluate(
            path=_p,
            sort_key="ID",
            save_to=_p,
            cal_metrics=False,
        )
        _beam_docode(args, _p)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config/baseline.yml", type=str, help="")
    args = parser.parse_args()
    args = read_yaml(args.config, to_args=True, **vars(args))
    args = combine_paths(args)
    return args


if __name__ == "__main__":
    args = make_args()
    set_seed()

    if args.only_test_mode:
        predict(args)
    else:
        # baseline lstm
        train(args)
        if args.test_mode:
            predict(args)
