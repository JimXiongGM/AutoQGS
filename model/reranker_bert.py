import logging
from typing import Dict, Optional

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.training.metrics.covariance import Covariance
from allennlp.training.metrics.mean_absolute_error import MeanAbsoluteError

from model.my_modeling_bert import BertModel

logger = logging.getLogger(__name__)


class TransformerRanker(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        model_name,
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self.linear_logit = torch.nn.Linear(768, 1, bias=False)
        self.bert = BertModel.from_pretrained(model_name)

        self.loss = torch.nn.MSELoss()
        self.covariance = Covariance()
        self.meanAbsoluteError = MeanAbsoluteError()

    def forward(  # type: ignore
        self,
        tokens: Dict[str, torch.Tensor],
        metadata,
        score: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        input_ids = tokens["tokens"]["token_ids"]
        input_mask = tokens["tokens"]["mask"]
        type_ids = tokens["tokens"]["type_ids"]

        outputs = {}
        outputs["metadata"] = metadata

        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=type_ids,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )

        logits = self.linear_logit(bert_outputs.pooler_output).squeeze()

        if score is not None:
            outputs["loss"] = self.loss(logits, score)

            if not self.training:
                self.covariance(predictions=logits, gold_labels=score)
                self.meanAbsoluteError(predictions=logits, gold_labels=score)
                pred_score = logits.tolist()
                outputs["pred_score"] = pred_score

        return outputs

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if not self.training:
            metrics["covariance"] = self.covariance.get_metric(reset)
            metrics.update(self.meanAbsoluteError.get_metric(reset))
        return metrics
