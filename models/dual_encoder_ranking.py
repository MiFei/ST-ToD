import os.path
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torch.nn import CrossEntropyLoss
from torch.nn import CosineEmbeddingLoss
from sklearn.metrics import f1_score 
import numpy as np


from transformers import *


class dual_encoder_ranking(nn.Module):
    def __init__(self, args): 
        super(dual_encoder_ranking, self).__init__()
        self.args = args
        self.xeloss = nn.CrossEntropyLoss()
        self.n_gpu = args["n_gpu"]

        ### Utterance Encoder
        self.utterance_encoder = args["model_class"].from_pretrained(self.args["model_name_or_path"])

        if self.args["fix_encoder"]:
            for p in self.utterance_encoder.parameters():
                p.requires_grad = False

        ## Prepare Optimizer
        def get_optimizer_grouped_parameters(model):
            param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
                 'lr': args["learning_rate"]},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                 'lr': args["learning_rate"]},
            ]
            return optimizer_grouped_parameters

        if self.n_gpu == 1:
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(self)
        else:
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(self.module)

        
        self.optimizer = AdamW(optimizer_grouped_parameters,
                                 lr=args["learning_rate"],)


    def optimize(self):
        self.loss_grad.backward()
        clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.args["grad_clip"])
        self.optimizer.step()

    def forward(self, data, ind_to_conf_map=None, evaluate_gradient=False, use_truth=True):

        if not evaluate_gradient:
            self.optimizer.zero_grad()

            batch_size = data["context"].size(0)

            interval = 25
            start_list = list(np.arange(0, batch_size, interval))
            end_list = start_list[1:] + [None]
            context_outputs, response_outputs = [], []

            for start, end in zip(start_list, end_list):

                inputs_con = {"input_ids": data["context"][start:end],
                              "attention_mask": (data["context"][start:end] > 0).long()}
                inputs_res = {"input_ids": data["response"][start:end],
                              "attention_mask": (data["response"][start:end] > 0).long()}

                if "bert" in self.args["model_type"]:
                    _, context_output = self.utterance_encoder(**inputs_con)
                    _, response_output = self.utterance_encoder(**inputs_res)
                elif self.args["model_type"] == "gpt2":
                    context_output = self.utterance_encoder(**inputs_con)[0].mean(1)
                    response_output = self.utterance_encoder(**inputs_res)[0].mean(1)
                elif self.args["model_type"] == "dialogpt":
                    transformer_outputs = self.utterance_encoder.transformer(**inputs_con)
                    context_output = transformer_outputs[0].mean(1)
                    transformer_outputs = self.utterance_encoder.transformer(**inputs_res)
                    response_output = transformer_outputs[0].mean(1)

                context_outputs.append(context_output.cpu())
                response_outputs.append(response_output.cpu())

            # evaluation for k-to-100
            if (not self.training) and (batch_size < 100):
                response_outputs.append(self.final_response_output[:100-batch_size, :])

            final_context_output = torch.cat(context_outputs, 0)
            final_response_output = torch.cat(response_outputs, 0)

            if torch.cuda.is_available():
                final_context_output = final_context_output.cuda()
                final_response_output = final_response_output.cuda()

            if not self.training:
                self.final_response_output = final_response_output.cpu()

            # mat
            logits = torch.matmul(final_context_output, final_response_output.transpose(1, 0))

            # loss
            labels = torch.tensor(np.arange(batch_size))
            if torch.cuda.is_available(): labels = labels.cuda()

            context_noised = final_context_output + 0.5 * torch.randn(final_context_output.size()).cuda()
            response_noised = final_response_output + 0.5 * torch.randn(final_response_output.size()).cuda()

            logits_context_noised = torch.matmul(context_noised, final_response_output.transpose(1, 0))
            logits_response_noised = torch.matmul(final_context_output, response_noised.transpose(1, 0))

            if self.args['embedding_noise']:  # add gaussian noise to embedding

                loss1 = self.xeloss(logits, labels)
                loss2 = self.xeloss(logits_context_noised, labels)

                loss = (loss1 + loss2) / 2

            else:

                weights = []
                for idx in data['index']:
                    if ind_to_conf_map and idx in ind_to_conf_map:
                        if self.args['confidence_weighting']:
                            weights.append(self.args['lambda'] * ind_to_conf_map[idx]) # weight samples based on confidence
                        else:
                            weights.append(self.args['lambda']) # weight pseudo labeled samples by lambda
                    else:
                        weights.append(1)
                weights = torch.FloatTensor(weights)
                if torch.cuda.is_available(): weights = weights.cuda()

                criterion = nn.CrossEntropyLoss(reduction='none')
                losses = criterion(logits, labels)

                weighted_loss = losses * weights
                loss = weighted_loss.mean()


            if self.training:
                self.loss_grad = loss
                self.optimize()

            predictions = np.argsort(logits.detach().cpu().numpy(), axis=1) #torch.argmax(logits, -1)

            softmax = nn.Softmax(-1)
            prob = softmax(logits)
            confidence, _ = torch.max(prob, dim=-1)

            outputs = {"loss":loss.item(),
                       "pred":predictions,
                       "label":np.arange(batch_size),
                       "prob": prob,
                       'confidence': confidence.detach().cpu().tolist()
                       }

            return outputs

        else:
            self.optimizer.zero_grad()

            batch_size = data["context"].size(0)

            interval = 25
            start_list = list(np.arange(0, batch_size, interval))
            end_list = start_list[1:] + [None]
            context_outputs, response_outputs = [], []

            for start, end in zip(start_list, end_list):

                inputs_con = {"input_ids": data["context"][start:end],
                              "attention_mask": (data["context"][start:end] > 0).long()}
                inputs_res = {"input_ids": data["response"][start:end],
                              "attention_mask": (data["response"][start:end] > 0).long()}

                _, context_output = self.utterance_encoder(**inputs_con)
                with torch.no_grad():
                    _, response_output = self.utterance_encoder(**inputs_res)

                context_outputs.append(context_output.cpu())
                response_outputs.append(response_output.cpu())

            final_context_output = torch.cat(context_outputs, 0)
            final_response_output = torch.cat(response_outputs, 0)

            if torch.cuda.is_available():
                final_context_output = final_context_output.cuda()
                final_response_output = final_response_output.cuda()

            # mat
            logits = torch.matmul(final_context_output, final_response_output.transpose(1, 0))

            # loss
            labels = torch.tensor(np.arange(batch_size))
            if torch.cuda.is_available():
                labels = labels.cuda()

            criterion = nn.CrossEntropyLoss(reduction='mean')
            losses = criterion(logits, labels)

            predictions = np.argsort(logits.detach().cpu().numpy(), axis=1)  # torch.argmax(logits, -1)

            losses.backward()

            softmax = nn.Softmax(-1)
            prob = softmax(logits)
            confidence, _ = torch.max(prob, dim=-1)

            outputs = {"loss": losses.item(),
                       "pred": predictions,
                       "label": np.arange(batch_size),
                       "prob": prob,
                       'confidence': confidence.detach().cpu().tolist(),
                       "logits": logits
                       }

            return outputs
    
    def evaluation(self, preds, labels):
        assert len(preds) == len(labels)
        
        preds = np.array(preds)
        labels = np.array(labels)
        
        def _recall_topk(preds_top10, labels, k):
            preds = preds_top10[:, -k:]
            acc = 0
            for li, label in enumerate(labels):
                if label in preds[li]: acc += 1
            acc = acc / len(labels)       
            return acc
        
        results = {"top-1": _recall_topk(preds, labels, 1), 
                   "top-3": _recall_topk(preds, labels, 3), 
                   "top-5": _recall_topk(preds, labels, 5), 
                   "top-10": _recall_topk(preds, labels, 10)}
        
        print(results)
        
        return results
