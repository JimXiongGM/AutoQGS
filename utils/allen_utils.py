import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
from allennlp.common.checks import check_for_gpu
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import dump_metrics, int_to_device, sanitize
from allennlp.data import DataLoader
from allennlp.models.model import Model
from allennlp.nn import util as nn_util

logger = logging.getLogger(__name__)


# We want to warn people that tqdm ignores metrics that start with underscores
# exactly once. This variable keeps track of whether we have.
class HasBeenWarned:
    tqdm_ignores_underscores = False


def get_metrics(
    model: Model,
    total_loss: float,
    total_reg_loss: Optional[float],
    batch_loss: Optional[float],
    batch_reg_loss: Optional[float],
    num_batches: int,
    reset: bool = False,
) -> Dict[str, float]:
    """
    Gets the metrics but sets `"loss"` to
    the total loss divided by the `num_batches` so that
    the `"loss"` metric is "average loss per batch".
    Returns the `"batch_loss"` separately.
    """
    metrics = model.get_metrics(reset=reset)
    if batch_loss is not None:
        metrics["batch_loss"] = batch_loss
    metrics["loss"] = float(total_loss / num_batches) if num_batches > 0 else 0.0
    if total_reg_loss is not None:
        if batch_reg_loss is not None:
            metrics["batch_reg_loss"] = batch_reg_loss
        metrics["reg_loss"] = float(total_reg_loss / num_batches) if num_batches > 0 else 0.0

    return metrics


def evaluate(
    model: Model,
    data_loader: DataLoader,
    cuda_device: Union[int, torch.device] = -1,
    batch_weight_key: str = None,
    output_file: str = None,
    predictions_output_file: str = None,
    total=None,
) -> Dict[str, Any]:
    """
    # Parameters

    model : `Model`
        The model to evaluate
    data_loader : `DataLoader`
        The `DataLoader` that will iterate over the evaluation data (data loaders already contain
        their data).
    cuda_device : `Union[int, torch.device]`, optional (default=`-1`)
        The cuda device to use for this evaluation.  The model is assumed to already be using this
        device; this parameter is only used for moving the input data to the correct device.
    batch_weight_key : `str`, optional (default=`None`)
        If given, this is a key in the output dictionary for each batch that specifies how to weight
        the loss for that batch.  If this is not given, we use a weight of 1 for every batch.
    metrics_output_file : `str`, optional (default=`None`)
        Optional path to write the final metrics to.
    predictions_output_file : `str`, optional (default=`None`)
        Optional path to write the predictions to.

    # Returns

    `Dict[str, Any]`
        The final metrics.
    """
    check_for_gpu(cuda_device)
    data_loader.set_target_device(int_to_device(cuda_device))
    predictions_file = (
        None if predictions_output_file is None else open(predictions_output_file, "w", encoding="utf-8")
    )

    with torch.no_grad():
        model.eval()

        iterator = iter(data_loader)
        logger.info("Iterating over dataset")
        generator_tqdm = Tqdm.tqdm(iterator, total=total, ncols=100)

        # Number of batches in instances.
        batch_count = 0
        # Number of batches where the model produces a loss.
        loss_count = 0
        # Cumulative weighted loss
        total_loss = 0.0
        # Cumulative weight across all batches.
        total_weight = 0.0

        for batch in generator_tqdm:
            batch_count += 1
            batch = nn_util.move_to_device(batch, cuda_device)
            output_dict = model(**batch)
            loss = output_dict.get("loss")

            metrics = model.get_metrics()

            if loss is not None:
                loss_count += 1
                if batch_weight_key:
                    weight = output_dict[batch_weight_key].item()
                else:
                    weight = 1.0

                total_weight += weight
                total_loss += loss.item() * weight
                # Report the average loss so far.
                metrics["loss"] = total_loss / total_weight

            if not HasBeenWarned.tqdm_ignores_underscores and any(
                metric_name.startswith("_") for metric_name in metrics
            ):
                logger.warning(
                    'Metrics with names beginning with "_" will ' "not be logged to the tqdm progress bar."
                )
                HasBeenWarned.tqdm_ignores_underscores = True
            description = (
                ", ".join(
                    [
                        "%s: %.2f" % (name, value)
                        for name, value in metrics.items()
                        if not name.startswith("_")
                    ]
                )
                + " ||"
            )
            generator_tqdm.set_description(description, refresh=False)

            if predictions_file is not None:
                predictions = json.dumps(sanitize(model.make_output_human_readable(output_dict)))
                predictions_file.write(predictions + "\n")

        if predictions_file is not None:
            predictions_file.close()

        final_metrics = model.get_metrics(reset=True)
        if loss_count > 0:
            # Sanity check
            if loss_count != batch_count:
                raise RuntimeError("The model you are trying to evaluate only sometimes produced a loss!")
            final_metrics["loss"] = total_loss / total_weight

        if output_file is not None:
            dump_metrics(output_file, final_metrics, log=True)

        return final_metrics
