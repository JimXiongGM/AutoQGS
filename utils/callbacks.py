import random
from copy import deepcopy
from typing import Any, Dict, List, Optional

import torch
from allennlp.data import TensorDict
from allennlp.training.callbacks.callback import TrainerCallback
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer

from common.common_utils import colorful, get_parameter_number, save_to_json, time_now
from predictor import Seq2SeqPredictor

# from allennlp_models.generation.predictors.seq2seq import Seq2SeqPredictor


class myEpochCallback(TrainerCallback):
    """
    调用： callback(self, metrics=metrics, epoch=epoch, is_master=self._master)
    """

    def __init__(self, reader=None, args=None):
        super().__init__(args.serialization_dir)
        self.reader = reader
        self.max_pred_on_epoch = args.max_pred_on_epoch if hasattr(args, "max_pred_on_epoch") else 0
        self.args = args
        self.logger = args.logger

        # not JSON serializable
        self._args = deepcopy(args.__dict__)
        if hasattr(args, "device"):
            self._args["device"] = repr(args.device)
        self._args["logger"] = repr(args.logger)

        self.start_time = time_now()
        self.best_epoch = -1
        self.parameter = None

        # logs
        self.logs = []

        # init instances
        self.test_instances = []
        if hasattr(args, "test_dataset") and args.test_dataset and self.reader:
            self.test_instances = [ins for ins in self.reader._read(args.test_dataset)]
            random.seed(0)
            random.shuffle(self.test_instances)
            self.test_instances = self.test_instances[: self.max_pred_on_epoch]

    def on_batch(
        self,
        trainer: "GradientDescentTrainer",
        batch_inputs: List[TensorDict],
        batch_outputs: List[Dict[str, Any]],
        batch_metrics: Dict[str, Any],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_primary: bool = True,
        batch_grad_norm: Optional[float] = None,
        **kwargs,
    ) -> None:
        """
        This callback hook is called after the end of each batch.
        """
        pass
        # if is_training:
        #     trainer.model.step_forcing_ratio()

    def on_epoch(
        self,
        trainer: "GradientDescentTrainer",
        metrics,
        epoch: int,
        is_primary: bool,
    ):
        if is_primary:
            # predictor = Seq2SeqPredictor(
            #     model=trainer.model, dataset_reader=self.reader
            # )
            if hasattr(trainer.model, "forcing_ratio"):
                print("forcing_ratio: ", trainer.model.forcing_ratio)
            # self.pred_on_epoch(predictor)
            if type(self.logs) == list and epoch >= 0:
                metrics["save_time"] = time_now()
                self.logs.append(deepcopy(metrics))

            if self.best_epoch != metrics["best_epoch"]:
                self.save_model(model=trainer.model)
                self.best_epoch = metrics["best_epoch"]

            print(
                colorful(
                    text=f"Current Epoch: {epoch+1}  Best: {self.best_epoch}",
                    color="yellow",
                )
            )
            self.save_logs(model=trainer.model, optimizer=trainer.optimizer)

    def pred_on_epoch(self, predictor):
        """
        should be overrided if need.
        """
        for instance in self.test_instances:
            print("SOURCE:")
            print("node: ", instance.fields["metadata"]["levi_nodes_name"])
            print("adj: ", instance.fields["metadata"]["levi_adj"])
            print("nodes_type: ", instance.fields["metadata"]["levi_nodes_type"])
            print(colorful("GOLD SEQ:", color="green"))
            print(instance.fields["metadata"]["ques_tokens"])
            print(colorful("PRED:", color="yellow"))
            print(predictor.predict_instance(instance)["predicted_tokens"])
            print()
            print("-*" * 50)
            print()

    def save_model(self, model, ver=None):
        if ver is not None:
            path = self.args.save_weights_file + f"-{ver}"
        else:
            path = self.args.save_weights_file
        torch.save(model.state_dict(), path)

    def save_logs(self, model, optimizer):
        if self.parameter is None:
            self.parameter = get_parameter_number(model, optimizer=optimizer)

        logs_info = {
            "args": self._args,
            "begin_time": self.start_time,
            "end_time": time_now(),
            "parameter": self.parameter,
            "details": sorted(self.logs, key=lambda x: x["save_time"]),
        }
        save_to_json(logs_info, self.args.log_name, _print=False)
