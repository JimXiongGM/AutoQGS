from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics import BLEU, ROUGE
from bert_BiGGNN.model.myTransformer import (
    Decoder,
    DecoderLayer,
    Embeddings,
    Generator,
    LabelSmoothing,
    MultiHeadedAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
    make_triu_mask_batch,
    subsequent_mask,
)
from transformers import BertModel, BertTokenizer


@Model.register("SequenceToSequence")
class SequenceToSequence(Model):
    def __init__(
        self,
        vocab=Vocabulary(
            padding_token="[PAD]",
            oov_token="[UNK]",
        ),
        transformer_model="datasets/pretrained_models/bert-base-uncased",
        tensor_based_metric=None,
        use_ROUGE=False,
        freeze_to_layer=None,
        **kwargs,
    ):
        super().__init__(vocab)
        self.transformer_model = transformer_model

        # model
        h = 8
        d_model = 768
        d_ff = 1024
        dropout = 0.1
        N = 6

        # src and tgt embedder
        self.encoder = BertModel.from_pretrained(transformer_model)
        self.source_tokenizer = BertTokenizer.from_pretrained(transformer_model)
        tgt_vocab_size = self.encoder.embeddings.word_embeddings.num_embeddings

        # Embeddings(num_embeddings=tgt_vocab_size, embedding_dim=d_model, padding_idx=0)
        # deepcopy(self.encoder.embeddings.word_embeddings)
        self.tgt_embed = nn.Sequential(
            Embeddings(num_embeddings=tgt_vocab_size, embedding_dim=d_model, padding_idx=0),
            PositionalEncoding(d_model, dropout),
        )

        # Encoder Decoder
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.decoder = Decoder(
            DecoderLayer(d_model, deepcopy(attn), deepcopy(attn), deepcopy(ff), dropout),
            N,
        )
        self.generator = Generator(d_model, tgt_vocab_size)

        # metric
        self.pad_index = 0
        self.start_index = 1
        self.end_index = 2
        self._tensor_based_metric = tensor_based_metric or BLEU(
            exclude_indices={self.pad_index, self.end_index, self.start_index}
        )
        self.max_len_tgt = 40
        self._token_based_metric = None

        if use_ROUGE:
            self._ROUGE = ROUGE(
                ngram_size=2,
                exclude_indices={self.pad_index, self.end_index, self.start_index},
            )
        else:
            self._ROUGE = None

        # self.criterion = LabelSmoothing(
        #     vocab_size=self.source_tokenizer.vocab_size,
        #     padding_idx=self.pad_index,
        #     smoothing=0.0,
        # )

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # pred
        self.method = "greedy"

        # freeze
        self._freeze_to_layer_by_name(layer_name=freeze_to_layer)

    def _freeze_to_layer_by_name(self, layer_name):
        """冻结层. 从0到layer_name."""
        if layer_name == None:
            return
        if layer_name == "all":
            index_start = len(self.encoder.state_dict())
        else:
            index_start = -1
            for index, (key, _value) in enumerate(self.encoder.state_dict().items()):
                if layer_name in key:
                    index_start = index
                    break

        if index_start < 0:
            print(f"Don't find layer name: {layer_name}")
            print(f"must in : \n{self.encoder.state_dict().keys()}")
            return

        no_grad_nums = index_start + 1
        grad_nums = 0

        for index, i in enumerate(self.encoder.parameters()):
            if index >= index_start:
                i.requires_grad = True
                grad_nums += 1
            else:
                i.requires_grad = False
        print(f"freeze layers num: {no_grad_nums}, active layers num: {grad_nums}.")

    def forward(
        self,
        src_input_ids,
        src_attention_mask,
        src_token_type_ids,
        target_input_ids_with_start=None,
        target_attention_mask_with_start=None,
        target_input_ids_with_end=None,
        metadata=None,
        **kwargs,
    ):
        """
        node text feature: for bert
            src_input_ids; src_attention_mask; src_token_type_ids: [bsz, src_num, tokennum]
        node feature:
            src_num: [bsz, 1]
        target:
            target_*: [bsz, tgt_len+1]
        """
        output_dict = {}
        output_dict["metadata"] = metadata

        # src_hidden: [bsz, src_num, dim]
        src_hidden = self.encode(src_input_ids, src_attention_mask, src_token_type_ids)

        src_attention_mask.unsqueeze_(1)
        if target_input_ids_with_start is not None:
            # hidden_states: [bsz, tgt_len, dim]
            # prob: [bsz, tgt_len, tgt_vocab]
            hidden_states, prob = self.decode(src_hidden, src_attention_mask, target_input_ids_with_start)
            # LabelSmoothing: [bsz * tgt_len, tgt_vocab] and [bsz * tgt_len]
            loss = self.criterion(
                prob.contiguous().view(-1, prob.size(-1)),
                target_input_ids_with_end.contiguous().view(-1),
            )
            output_dict["loss"] = loss

        # pred or eval
        if not self.training:
            # pred_tgt, pred_probs: tensor [bsz, tgt_len]
            pred_tgt, pred_probs = self._forward_search(memory=src_hidden, src_mask=src_attention_mask)
            output_dict["pred_tgt"] = pred_tgt
            output_dict["pred_probs"] = pred_probs

            # eval
            if target_input_ids_with_start is not None:
                if self._tensor_based_metric is not None:
                    # pred_tgt: [bsz, tgt_pred_len]
                    # gold_tgt: [bsz, tgt_gold_len]
                    self._tensor_based_metric(pred_tgt, target_input_ids_with_end)

                if self._token_based_metric is not None:
                    predicted_tokens = self._get_predicted_tokens(output_dict["pred_tgt"])
                    self._token_based_metric(  # type: ignore
                        predicted_tokens, [x["target_tokens"] for x in metadata]
                    )

        return output_dict

    def _greedy_decode(self, memory, src_mask):
        """
        inputs
            memory: [bsz, src_len, dim]
            src_mask: [bsz, 1, src_len]
        return
            pred_tgt, pred_probs: [bsz, tgt_len]
        """
        bsz = memory.shape[0]
        pred_tgt = torch.ones(bsz, 1).fill_(self.start_index).long().to(memory.device)
        pred_probs = None
        for i in range(self.max_len_tgt - 1):
            # 下三角矩阵 逐渐增大
            hidden_states, prob = self.decode(
                encoder_output=memory,
                src_mask=src_mask,
                tgt=pred_tgt,
                tgt_mask=subsequent_mask(pred_tgt.size(1)).type_as(memory.data),
            )
            prob = prob[:, -1, :]
            next_prob, next_word = torch.max(prob, dim=1)
            pred_tgt = torch.cat([pred_tgt, next_word.type_as(pred_tgt).unsqueeze(-1)], dim=-1)
            pred_probs = (
                torch.cat([pred_probs, next_prob.type_as(pred_probs).unsqueeze(-1)], dim=-1)
                if pred_probs is not None
                else next_prob.unsqueeze(-1)
            )
        return pred_tgt, pred_probs

    def _forward_search(self, memory, src_mask):
        """
        greedy or beam search
        """
        if self.method == "greedy":
            pred_tgt, pred_probs = self._greedy_decode(memory, src_mask)
            return pred_tgt, pred_probs
        elif self.method == "beam":
            pass
        else:
            raise ValueError(f"wrong. {self.method} not in `[greedy, beam]`")

    def _get_predicted_tokens(self, pred_tgt):
        """
        pred_tgt: tensor, [tgt_len]
        """
        predicted_tokens = self.source_tokenizer.batch_decode(pred_tgt)
        return predicted_tokens

    def encode(self, src_input_ids, src_attention_mask, src_token_type_ids):
        """
        Encode source input graphs
        """
        _inputs = {
            "input_ids": src_input_ids,
            "attention_mask": src_attention_mask,
            "token_type_ids": src_token_type_ids,
        }
        bert_out = self.encoder(**_inputs)
        return bert_out.last_hidden_state

    def decode(self, encoder_output, src_mask, tgt, tgt_mask=None):
        """
        input
            encoder_output: [bsz, src_len, dim]
            src_mask: [bsz, 1, src_len]
            tgt: [bsz, tgt_len]
            tgt_mask: [bsz, tgt_len] -> triu [bsz, tgt_len, tgt_len]
        return
            hidden_states: [bsz, tgt_len, dim]
            prob: [bsz, tgt_len, tgt_vocab]
        """
        tgt_mask = make_triu_mask_batch(tgt, pad=0) if tgt_mask is None else tgt_mask
        hidden_states = self.decoder(self.tgt_embed(tgt), encoder_output, src_mask, tgt_mask)
        prob = self.generator(hidden_states)
        return hidden_states, prob

    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]):
        """
        Finalize predictions.

        After a beam search, the predicted indices correspond to tokens in the target vocabulary
        OR tokens in source sentence. Here we gather the actual tokens corresponding to
        the indices.
        """
        # list of [str]
        predicted_tokens = self._get_predicted_tokens(output_dict["pred_tgt"])
        # allennlp requires
        output_dict["predicted_tokens"] = predicted_tokens
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            if self._tensor_based_metric is not None:
                all_metrics.update(self._tensor_based_metric.get_metric(reset=reset))  # type: ignore

            if self._ROUGE:
                all_metrics.update(self._ROUGE.get_metric(reset=reset))

            if self._token_based_metric is not None:
                all_metrics.update(self._token_based_metric.get_metric(reset=reset))  # type: ignore
        return all_metrics


if __name__ == "__main__":
    batch = 5
    src_num = 6
    nodetoken = 7
    dim = 512
    tgt_len = 19
    src_tkids = torch.randint(102, 999, size=(batch, src_num, nodetoken)).long()
    src_valid_num = torch.randint(0, 2, size=(batch,))
    target_ids = torch.randint(102, 999, size=(batch, tgt_len))
    target_mask = torch.randint(0, 19, size=(batch,))

    model = SequenceToSequence()
